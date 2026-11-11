"""Shared per-robot target estimator: Kalman filter + innovation gating + belief reset.

Used identically by all three strategies (independent / centralized / decentralized),
which then differ only in how information is shared.  See SHARED_COMPONENTS.md.

Model
-----
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

``gate`` is a chi-square threshold with 2 degrees of freedom; for 2 dof the
quantile has a closed form, ``chi2_threshold(p) = -2 ln(1 - p)`` (p = 0.99 -> 9.21).
Sweeping ``gate`` trades false resets (noise taken for a relocation) against
missed relocations (a real move averaged away).  Every gating decision is
appended to ``Estimator.decisions`` so those rates can be measured.
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


@dataclass
class Estimator:
    """Holds the shared settings and the gating log; the beliefs themselves live in
    ``robot.beliefs`` so every strategy reads and writes the same ``Belief`` objects."""

    config: EstimatorConfig = field(default_factory=EstimatorConfig)
    decisions: List[GateDecision] = field(default_factory=list)
    _pending_outliers: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _last_reset: Dict[Tuple[int, int], bool] = field(default_factory=dict)

    def reset_log(self) -> None:
        self.decisions.clear()
        self._pending_outliers.clear()
        self._last_reset.clear()

    # -- building blocks (pure functions of a Belief) ------------------------
    def predict(self, belief: Belief, t: float) -> Belief:
        """Inflate the covariance for the time since the belief was last updated."""
        if not belief.valid or belief.cov is None or not np.isfinite(belief.timestamp):
            return belief
        dt = max(0.0, t - belief.timestamp)
        return replace(belief, cov=belief.cov + self.config.process_noise * dt * np.eye(2), timestamp=t)

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
    def update(self, robot_id: int, target_id: int, belief: Belief, z: Optional[np.ndarray],
               sensor_noise_std: float, t: float) -> Belief:
        """Predict to time ``t`` and, if there is an observation, gate and update/reset.

        Call once per robot, target and step.  Afterwards ``relocation_detected``
        tells whether this call reset the belief."""
        key = (robot_id, target_id)
        self._last_reset[key] = False
        belief = self.predict(belief, t)
        if z is None:
            return belief
        z = np.asarray(z, dtype=float)
        R = (sensor_noise_std ** 2) * np.eye(2)
        if not belief.valid or belief.cov is None:
            return Belief(mean=z.copy(), cov=R.copy(), timestamp=t, source="own_obs", last_seen=t,
                          reset_count=belief.reset_count)

        _, _, d2 = self.innovation(belief, z, R)
        outlier = d2 > self.config.gate
        reset = False
        if outlier:
            self._pending_outliers[key] = self._pending_outliers.get(key, 0) + 1
            if self._pending_outliers[key] >= self.config.confirm:
                reset = True
                self._pending_outliers[key] = 0
                belief = Belief(mean=z.copy(), cov=R.copy(), timestamp=t, source="reset", last_seen=t,
                                reset_count=belief.reset_count + 1)
            # an unconfirmed outlier is neither fused nor allowed to reset yet
        else:
            self._pending_outliers[key] = 0
            belief = self.kalman_update(belief, z, R, t)
        self._last_reset[key] = reset
        self.decisions.append(GateDecision(t, robot_id, target_id, d2, self.config.gate, outlier, reset))
        return belief

    def relocation_detected(self, robot_id: int, target_id: int) -> bool:
        """True if the latest ``update`` for this robot and target reset the belief."""
        return self._last_reset.get((robot_id, target_id), False)
