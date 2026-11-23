"""Shared per-robot target estimator: Kalman filter + innovation gating + belief reset.

Used identically by all three strategies (independent / centralized / decentralized),
which then differ only in how information is shared.  See SHARED_COMPONENTS.md.

Two models (``EstimatorConfig.model``)
-------------------------------------
* ``"imm"`` (default): the target is either RESTING or being CARRIED by a
  person.  An Interacting Multiple Model filter runs one Kalman filter per mode
  on a shared [x, y, vx, vy] state -- resting: velocity forced to zero;
  carried: constant velocity with white-noise acceleration -- and keeps the
  probability of each mode (``Belief.p_moving``).  On steps without a
  sighting the carried velocity decays with ``velocity_decay_time``
  (Singer-style), so an unseen carry coasts a few metres rather than
  indefinitely; with a sighting it is plain constant velocity.  Each step: mix the two
  filters according to the switch probabilities (from the mean rest and walk
  durations), predict each, update each with the sighting, reweight the modes by
  how well each predicted it.  ``Belief.mean`` / ``cov`` are the blended
  position, so everything downstream is unchanged.  Gating, reset and the
  no-sighting rule work as below, with the gate applied to the best-fitting
  mode: a sighting is an outlier only if NEITHER model explains it (e.g. the
  target was carried out of view and turns up elsewhere).
  If something outside the estimator changes ``Belief.mean`` / ``cov`` (e.g. a
  consensus step), the next update re-centres both modes on the new position
  and covariance, keeping the velocity and mode probabilities.
* ``"static"``: the original single filter described below (ablation).

Static model
------------
The target is static between relocations, so the state is its 2D position.

* predict: the mean stays, the covariance grows by ``process_noise * dt * I``.
  This is not motion; it encodes "the longer since we looked, the less sure we
  are that it is still there", so old beliefs lose against fresh information.
* update with an observation z (covariance R = sensor_noise_std^2 * I):
  innovation  nu = z - mean,   S = P + R,   d2 = nu^T S^-1 nu   (Mahalanobis^2)
  - d2 <= gate:  normal Kalman update (noise is averaged away);
  - d2 >  gate:  the observation does not fit the belief.  After ``confirm``
                 consecutive such observations the target is declared relocated
                 and the belief is RESET to the latest observation instead of
                 being averaged with the old position.  Outliers before that are
                 not fused.
  With no prior belief the first observation initialises it (not a reset).
* no observation, but the believed position was in view (decision 3, the
  no-sighting rule): the sensor never misses a target in range, so the target
  is not where we thought.  Because the target teleports to a random spot, the
  old position says nothing about the new one, so the belief is CLEARED (no
  mean, no covariance) -- "I don't know where it is, except not here".  The
  check itself is kept: ``last_absence`` (when), ``absence_pos`` and
  ``absence_radius`` (the area seen empty).  A neighbour's belief that points
  into that area and rests on an older sighting is contradicted
  (``messages.contradicted_by_absence``), so a stale rumour cannot re-infect the
  robot.  ``last_seen`` is not changed.  The next sighting starts a fresh belief
  (not counted as a relocation reset).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .robot import Belief


def chi2_threshold(p: float, dof: int = 2) -> float:
    """Chi-square quantile; closed form for the 2-dof case used here."""
    if dof != 2:
        raise ValueError("only the 2-dof (planar position) case is needed")
    return float(-2.0 * np.log(1.0 - p))


@dataclass
class GateDecision:
    """One gating decision, for the false-reset / missed-relocation metrics."""
    t: float
    robot_id: int
    target_id: int
    d2: float           # Mahalanobis^2 of the innovation
    gate: float         # threshold used
    outlier: bool       # d2 > gate
    reset: bool         # the belief was reset (relocation declared)


@dataclass
class EstimatorConfig:
    process_noise: float = 0.01          # m^2/s added to each variance per second
    gate: float = chi2_threshold(0.99)   # chi-square(2) threshold on d2  (sweep this)
    confirm: int = 1                     # consecutive outliers needed before a reset
    min_cov: float = 1e-6                # floor on the variances, for numerical safety
    absence: bool = True                 # no-sighting rule on/off
    absence_view_fraction: float = 0.8   # believed position must be within this share of sensor range
    # resting / carried (IMM) model
    model: str = "imm"                   # "imm" | "static"
    mean_rest: float = 60.0              # s, expected time the target rests between moves
    mean_walk: float = 10.0              # s, expected duration of a carry
    accel_noise: float = 1.5             # m/s^2, std of the carrier's acceleration (sharp turns at doors, starts, stops)
    init_speed_std: float = 0.85         # m/s per axis, carried-mode velocity std at (re)initialisation and
                                         # its cap (walking at 1.2 m/s in a random direction: 1.2/sqrt(2))
    max_pos_std: float = 6.0             # m, cap on the position std of each mode (about the building's spread)
    velocity_decay_time: float = 2.0     # s, on steps WITHOUT a sighting the carried mode's velocity decays
                                         # with this time constant (Singer-style), so an unseen carry coasts
                                         # about speed * this far instead of straight through walls; with a
                                         # sighting it is plain constant velocity (no bias toward zero)
    rest_velocity_var: float = 1e-4      # (m/s)^2, velocity variance kept in the resting mode

    def __post_init__(self) -> None:
        if self.model not in ("imm", "static"):
            raise ValueError(f"unknown estimator model {self.model!r}; expected 'imm' or 'static'")


@dataclass
class Estimator:
    """Holds the shared settings and the gating log; the beliefs themselves live in
    ``robot.beliefs`` so every strategy reads and writes the same ``Belief`` objects."""

    config: EstimatorConfig = field(default_factory=EstimatorConfig)
    decisions: List[GateDecision] = field(default_factory=list)
    _pending_outliers: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _last_reset: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    absences: List[Tuple[float, int, int]] = field(default_factory=list)   # (t, robot_id, target_id)
    _last_absence: Dict[Tuple[int, int], bool] = field(default_factory=dict)

    def reset_log(self) -> None:
        self.decisions.clear()
        self._pending_outliers.clear()
        self._last_reset.clear()
        self.absences.clear()
        self._last_absence.clear()

    # -- building blocks (pure functions of a Belief) ------------------------
    def predict(self, belief: Belief, t: float, sighted: bool = False) -> Belief:
        """Static model: inflate the covariance for the time since the belief was
        last updated.  IMM model: mix and predict both modes to time ``t``
        (``sighted``: a sighting follows, so the carried velocity is not decayed)."""
        if not belief.valid or belief.cov is None or not np.isfinite(belief.timestamp):
            return belief
        if self.config.model == "imm":
            return self._imm_predict(belief, t, sighted)
        dt = max(0.0, t - belief.timestamp)
        return replace(belief, cov=belief.cov + self.config.process_noise * dt * np.eye(2), timestamp=t)


    # -- resting / carried (IMM) model ---------------------------------------
    # Mode 0 = resting, mode 1 = carried.  State x = [px, py, vx, vy].
    _H = np.hstack([np.eye(2), np.zeros((2, 2))])

    def _imm_init(self, z: np.ndarray, R: np.ndarray, p_moving: Optional[float] = None) -> dict:
        c = self.config
        x = np.concatenate([z, np.zeros(2)])
        P_rest = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), c.rest_velocity_var * np.eye(2)]])
        P_carr = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), c.init_speed_std ** 2 * np.eye(2)]])
        if p_moving is None:
            p_moving = c.mean_walk / (c.mean_walk + c.mean_rest)   # long-run share of time carried
        return {"x": [x.copy(), x.copy()], "P": [P_rest, P_carr], "mu": np.array([1.0 - p_moving, p_moving]),
                "out": (z.copy(), R.copy())}

    def _imm_state(self, belief: Belief) -> dict:
        """The belief's IMM state, created or re-centred if the blended position
        was changed from outside (a consensus step or an adopted message)."""
        st = belief.imm
        if st is None:
            st = self._imm_init(np.asarray(belief.mean, float), np.asarray(belief.cov, float))
            if belief.velocity is not None:
                st["x"][1][2:] = belief.velocity
            return st
        mean_out, cov_out = st["out"]
        if np.allclose(mean_out, belief.mean) and np.allclose(cov_out, belief.cov):
            return st
        st = {"x": [x.copy() for x in st["x"]], "P": [P.copy() for P in st["P"]], "mu": st["mu"].copy()}
        for j in range(2):
            st["x"][j][:2] = belief.mean
            st["P"][j][:2, :2] = belief.cov
            st["P"][j][:2, 2:] = 0.0
            st["P"][j][2:, :2] = 0.0
        return st

    def _switch_matrix(self, dt: float) -> np.ndarray:
        """Pi[i, j] = P(mode j now | mode i dt ago)."""
        c = self.config
        p_go = 1.0 - np.exp(-dt / c.mean_rest)
        p_stop = 1.0 - np.exp(-dt / c.mean_walk)
        return np.array([[1.0 - p_go, p_go], [p_stop, 1.0 - p_stop]])

    def _models(self, dt: float, decay: bool = True):
        c = self.config
        I2, Z2 = np.eye(2), np.zeros((2, 2))
        F_rest = np.block([[I2, Z2], [Z2, Z2]])
        Q_rest = np.block([[c.process_noise * dt * I2, Z2], [Z2, c.rest_velocity_var * I2]])
        tau = c.velocity_decay_time
        if decay and np.isfinite(tau):
            a = np.exp(-dt / tau)
            F_carr = np.block([[I2, tau * (1.0 - a) * I2], [Z2, a * I2]])
        else:
            F_carr = np.block([[I2, dt * I2], [Z2, I2]])
        q = c.accel_noise ** 2
        Q_carr = q * np.block([[dt ** 4 / 4 * I2, dt ** 3 / 2 * I2], [dt ** 3 / 2 * I2, dt ** 2 * I2]])
        Q_carr[:2, :2] += c.process_noise * dt * I2
        return (F_rest, Q_rest), (F_carr, Q_carr)

    def _cap(self, P: np.ndarray) -> np.ndarray:
        """Bound the uncertainty to what is physical: speed no more uncertain than
        walking pace, position no more uncertain than the building.  Scales rows
        and columns (P <- D P D), which keeps P positive semi-definite."""
        c = self.config
        caps = np.array([c.max_pos_std ** 2] * 2 + [c.init_speed_std ** 2] * 2)
        f = np.sqrt(np.minimum(1.0, caps / np.maximum(np.diag(P), 1e-300)))
        return P * np.outer(f, f)

    @staticmethod
    def _combine(xs, Ps, w):
        x = sum(wi * xi for wi, xi in zip(w, xs))
        P = sum(wi * (Pi + np.outer(xi - x, xi - x)) for wi, xi, Pi in zip(w, xs, Ps))
        return x, 0.5 * (P + P.T)

    def _imm_output(self, belief: Belief, st: dict, t: float, **kw) -> Belief:
        x, P = self._combine(st["x"], st["P"], st["mu"])
        mean, cov = x[:2].copy(), P[:2, :2] + self.config.min_cov * np.eye(2)
        st = dict(st, out=(mean.copy(), cov.copy()))
        return replace(belief, mean=mean, cov=cov, velocity=x[2:].copy(), p_moving=float(st["mu"][1]),
                       imm=st, timestamp=t, **kw)

    def _imm_predict(self, belief: Belief, t: float, sighted: bool = False) -> Belief:
        st = self._imm_state(belief)
        dt = max(0.0, t - belief.timestamp)
        if dt == 0.0:
            return self._imm_output(belief, st, t)
        Pi = self._switch_matrix(dt)
        mu = st["mu"]
        c_j = mu @ Pi                                     # predicted mode probabilities
        w = (Pi * mu[:, None]) / np.maximum(c_j, 1e-300)  # w[i, j] = P(was i | now j)
        xs, Ps = [], []
        for j, (F, Q) in enumerate(self._models(dt, decay=not sighted)):
            x0, P0 = self._combine(st["x"], st["P"], w[:, j])
            xs.append(F @ x0)
            Ps.append(self._cap(F @ P0 @ F.T + Q))
        return self._imm_output(belief, {"x": xs, "P": Ps, "mu": c_j / c_j.sum()}, t)

    def _imm_gate(self, belief: Belief, z: np.ndarray, R: np.ndarray):
        """Per-mode (nu, S, d2) of ``z``; the gate uses the best-fitting mode."""
        st = self._imm_state(belief)
        out = []
        for x, P in zip(st["x"], st["P"]):
            nu = z - x[:2]
            S = P[:2, :2] + R
            out.append((nu, S, float(nu @ np.linalg.solve(S, nu))))
        return st, out

    def _imm_update(self, belief: Belief, st: dict, per_mode, R: np.ndarray, t: float) -> Belief:
        xs, Ps, logL = [], [], []
        for (x, P), (nu, S, d2) in zip(zip(st["x"], st["P"]), per_mode):
            K = P @ self._H.T @ np.linalg.inv(S)
            xs.append(x + K @ nu)
            Pn = (np.eye(4) - K @ self._H) @ P
            Ps.append(0.5 * (Pn + Pn.T))
            logL.append(-0.5 * d2 - 0.5 * np.log(np.linalg.det(2 * np.pi * S)))
        logw = np.log(np.maximum(st["mu"], 1e-300)) + np.array(logL)
        mu = np.exp(logw - logw.max())
        mu = np.clip(mu / mu.sum(), 1e-6, 1.0)
        return self._imm_output(belief, {"x": xs, "P": Ps, "mu": mu / mu.sum()}, t, last_seen=t, source="own_obs")

    def _fresh(self, belief: Belief, z: np.ndarray, R: np.ndarray, t: float, **kw) -> Belief:
        """A belief started from one sighting (first sighting or reset)."""
        if self.config.model == "imm":
            return self._imm_output(belief, self._imm_init(z, R), t, last_seen=t, **kw)
        return replace(belief, mean=z.copy(), cov=R.copy(), timestamp=t, last_seen=t, **kw)

    @staticmethod
    def innovation(belief: Belief, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """(nu, S, d2) of observation ``z`` against ``belief``."""
        nu = np.asarray(z, dtype=float) - belief.mean
        S = belief.cov + R
        return nu, S, float(nu @ np.linalg.solve(S, nu))

    def kalman_update(self, belief: Belief, z: np.ndarray, R: np.ndarray, t: float) -> Belief:
        nu, S, _ = self.innovation(belief, z, R)
        K = belief.cov @ np.linalg.inv(S)
        cov = (np.eye(2) - K) @ belief.cov
        cov = 0.5 * (cov + cov.T) + self.config.min_cov * np.eye(2)
        return replace(belief, mean=belief.mean + K @ nu, cov=cov, timestamp=t, last_seen=t, source="own_obs")

    # -- the full step every strategy calls ----------------------------------
    @staticmethod
    def apply_absence(belief: Belief, t: float, view_pos: np.ndarray, view_radius: float) -> Belief:
        """The no-sighting rule on its own: forget the position, remember the
        check (when, and the area seen empty)."""
        return Belief(timestamp=t, source="absence", last_seen=belief.last_seen,
                      reset_count=belief.reset_count, last_absence=t,
                      absence_pos=np.array(view_pos, dtype=float), absence_radius=float(view_radius))

    def update(self, robot_id: int, target_id: int, belief: Belief, z: Optional[np.ndarray],
               sensor_noise_std: float, t: float, in_view: bool = False,
               view_pos: Optional[np.ndarray] = None, view_radius: float = 0.0) -> Belief:
        """Predict to time ``t``; with an observation, gate and update/reset;
        without one, apply the no-sighting rule if the believed position was
        ``in_view`` (the caller decides, see ``_SharedEstimatorBase.belief_in_view``);
        ``view_pos`` / ``view_radius`` describe the area the robot saw empty.

        Call once per robot, target and step.  Afterwards ``relocation_detected``
        / ``absence_detected`` tell whether this call reset the belief or found
        the expected spot empty."""
        key = (robot_id, target_id)
        self._last_reset[key] = False
        self._last_absence[key] = False
        belief = self.predict(belief, t, sighted=z is not None)
        if z is None:
            if in_view and self.config.absence and belief.valid:
                belief = self.apply_absence(belief, t, view_pos, view_radius)
                self._pending_outliers[key] = 0
                self._last_absence[key] = True
                self.absences.append((t, robot_id, target_id))
            return belief
        z = np.asarray(z, dtype=float)
        R = max(sensor_noise_std ** 2, self.config.min_cov) * np.eye(2)
        if not belief.valid or belief.cov is None:
            return self._fresh(belief, z, R, t, source="own_obs")

        if self.config.model == "imm":
            st, per_mode = self._imm_gate(belief, z, R)
            d2 = min(m[2] for m in per_mode)
        else:
            _, _, d2 = self.innovation(belief, z, R)
        outlier = d2 > self.config.gate
        reset = False
        if outlier:
            self._pending_outliers[key] = self._pending_outliers.get(key, 0) + 1
            if self._pending_outliers[key] >= self.config.confirm:
                reset = True
                self._pending_outliers[key] = 0
                belief = self._fresh(belief, z, R, t, source="reset", reset_count=belief.reset_count + 1)
            # an unconfirmed outlier is neither fused nor allowed to reset yet
        else:
            self._pending_outliers[key] = 0
            if self.config.model == "imm":
                belief = self._imm_update(belief, st, per_mode, R, t)
            else:
                belief = self.kalman_update(belief, z, R, t)
        self._last_reset[key] = reset
        self.decisions.append(GateDecision(t, robot_id, target_id, d2, self.config.gate, outlier, reset))
        return belief

    def relocation_detected(self, robot_id: int, target_id: int) -> bool:
        """True if the latest ``update`` for this robot and target reset the belief."""
        return self._last_reset.get((robot_id, target_id), False)

    def absence_detected(self, robot_id: int, target_id: int) -> bool:
        """True if the latest ``update`` found the believed position empty."""
        return self._last_absence.get((robot_id, target_id), False)
