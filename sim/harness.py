"""Experiment harness: config, simulation loop, trial runner, logging."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .comms import CommunicationLayer
from .dispatch import Dispatcher, DispatchResult
from .environment import Environment
from .faults import FaultSchedule
from .planning import plan_path
from .robot import Belief, Robot
from .strategies import CoordinationStrategy, make_strategy
from .target import Target


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    name: str = "default"
    # world
    floor_plan: Optional[str] = None       # None=default layout, else path to .npy/.png/.txt
    resolution: float = 0.25
    robot_radius: float = 0.3
    # robots
    n_robots: int = 3
    max_speed: float = 0.6
    max_accel: float = 0.8
    sensor_range: float = 3.0
    comm_range: float = 6.0
    sensor_noise_std: float = 0.3
    use_line_of_sight: bool = False
    kinematics: str = "holonomic"
    patrol: bool = True                    # robots wander to random free goals when idle
    # targets
    n_targets: int = 1
    relocation_dist: str = "uniform"       # uniform | exponential | fixed | none
    relocation_min: float = 30.0
    relocation_max: float = 90.0
    relocation_mean: float = 60.0
    # strategy
    strategy: str = "independent"
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    # faults
    packet_loss_prob: float = 0.0
    robot_failures: Dict[int, float] = field(default_factory=dict)  # robot_id -> t
    # queries / dispatch
    query_times: List[float] = field(default_factory=list)   # explicit query times
    query_interval: Optional[float] = None                  # or periodic queries
    first_query_at: float = 30.0
    dispatch_success_radius: float = 1.0
    dispatch_timeout: float = 120.0
    # trial
    dt: float = 0.1
    time_limit: float = 300.0
    n_trials: int = 100
    seed: int = 0
    localised_threshold: float = 1.0   # belief error below this counts as "localised"
    log_every: int = 1                  # log every k steps
    output_dir: str = "results"

    # -- helpers ------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.robot_failures = {int(k): float(v) for k, v in cfg.robot_failures.items()}
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "SimConfig":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        return cls.from_dict(d or {})

    def with_overrides(self, **kw) -> "SimConfig":
        d = asdict(self)
        d.update(kw)
        return SimConfig.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_environment(cfg: SimConfig) -> Environment:
    fp = cfg.floor_plan
    if fp is None:
        return Environment(resolution=cfg.resolution, robot_radius=cfg.robot_radius)
    if fp.endswith(".npy"):
        return Environment(grid=np.load(fp), resolution=cfg.resolution, robot_radius=cfg.robot_radius)
    if fp.endswith(".txt"):
        return Environment.from_array(np.loadtxt(fp, dtype=int), resolution=cfg.resolution, robot_radius=cfg.robot_radius)
    return Environment.from_image(fp, resolution=cfg.resolution, robot_radius=cfg.robot_radius)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class Simulation:
    """One trial. Build, then call ``step()`` repeatedly or ``run()``."""

    def __init__(self, cfg: SimConfig, seed: Optional[int] = None, env: Optional[Environment] = None,
                 strategy: Optional[CoordinationStrategy] = None):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.env = env if env is not None else load_environment(cfg)
        self.t = 0.0
        self.step_idx = 0

        # targets
        self.targets: List[Target] = []
        for k in range(cfg.n_targets):
            pos = self.env.random_free_position(self.rng)
            tg = Target(id=k, position=pos, interval_dist=cfg.relocation_dist, interval_min=cfg.relocation_min,
                        interval_max=cfg.relocation_max, interval_mean=cfg.relocation_mean)
            tg.schedule_next(0.0, self.rng)
            self.targets.append(tg)

        # robots: random collision-free, spread apart
        self.robots: List[Robot] = []
        for i in range(cfg.n_robots):
            pos = self.env.random_free_position(self.rng, min_dist_from=[r.position for r in self.robots],
                                                min_dist=2 * cfg.robot_radius + 0.2)
            self.robots.append(Robot(id=i, position=pos, heading=self.rng.uniform(-np.pi, np.pi),
                                     max_speed=cfg.max_speed, max_accel=cfg.max_accel,
                                     sensor_range=cfg.sensor_range,
                                     comm_range=cfg.comm_range, sensor_noise_std=cfg.sensor_noise_std,
                                     kinematics=cfg.kinematics))

        self.comm = CommunicationLayer(packet_loss_prob=cfg.packet_loss_prob, rng=self.rng)
        self.faults = FaultSchedule(robot_failures=dict(cfg.robot_failures), packet_loss_prob=cfg.packet_loss_prob)
        self.strategy = strategy if strategy is not None else make_strategy(cfg.strategy, **cfg.strategy_kwargs)
        self.strategy.reset(self.robots, self.rng)
        self.dispatcher = Dispatcher(self.env, success_radius=cfg.dispatch_success_radius, timeout=cfg.dispatch_timeout)

        # query schedule
        self._pending_queries = sorted(cfg.query_times)
        self._next_periodic = cfg.first_query_at if cfg.query_interval else None

        # logs
        self.step_log: List[dict] = []
        self.events: List[dict] = []

    # ---- main loop ---------------------------------------------------------
    def step(self) -> None:
        cfg, t, dt = self.cfg, self.t, self.cfg.dt

        # 1. faults
        self.faults.apply(t, self.robots)

        # 2. targets relocate
        for tg in self.targets:
            ev = tg.step(t, self.env, self.rng)
            if ev is not None:
                self.events.append({"t": t, "event": "relocation", "target_id": tg.id,
                                    "x": ev.new_pos[0], "y": ev.new_pos[1]})

        # 3. comm graph (stub)
        self.comm.build_graph(self.robots, t)

        # 4. sense + belief update (stub) + relocation detection (stub)
        for r in self.robots:
            obs = r.sense(self.targets, self.env, self.rng, cfg.use_line_of_sight)
            msgs = r.inbox
            r.inbox = []
            if r.failed:
                continue
            for tg in self.targets:
                b = self.strategy.update_belief(r, tg.id, obs[tg.id], msgs, t)
                r.beliefs[tg.id] = b
                if self.strategy.detect_relocation(r, tg.id, obs[tg.id], b):
                    self.events.append({"t": t, "event": "relocation_detected", "robot_id": r.id, "target_id": tg.id})

        # 5. messaging (stub)
        for r in self.robots:
            if r.failed:
                continue
            for receiver_id, payload in self.strategy.get_messages_to_send(r, self.comm, self.robots, t):
                if receiver_id is None:
                    self.comm.broadcast(r, self.robots, payload)
                else:
                    self.comm.send_message(r, self.robots[receiver_id], payload)

        # 6. queries
        self._issue_queries(t)

        # 7. dispatch bookkeeping
        for res in self.dispatcher.step(t, self.robots):
            self.events.append({"t": t, "event": "dispatch_end", "query_id": res.query_id, "robot_id": res.robot_id,
                                "target_id": res.target_id, "success": res.success, "reason": res.reason})

        # 8. motion (patrol planning for idle robots)
        for r in self.robots:
            if r.failed:
                continue
            if cfg.patrol and r.mode == "patrol" and not r.has_path:
                goal = self.env.random_free_position(self.rng)
                r.set_path(plan_path(r.position, goal, self.env))
            r.step(dt, self.env)

        # 9. log
        if self.step_idx % cfg.log_every == 0:
            self._log_step()

        self.step_idx += 1
        self.t = round(self.step_idx * dt, 9)

    def run(self, time_limit: Optional[float] = None, callback=None) -> None:
        tl = self.cfg.time_limit if time_limit is None else time_limit
        while self.t < tl:
            self.step()
            if callback is not None:
                callback(self)

    # ---- internals ---------------------------------------------------------
    def _issue_queries(self, t: float) -> None:
        due = False
        while self._pending_queries and self._pending_queries[0] <= t:
            self._pending_queries.pop(0)
            due = True
        if self._next_periodic is not None and t >= self._next_periodic:
            self._next_periodic += self.cfg.query_interval
            due = True
        if due:
            for tg in self.targets:
                res = self.dispatcher.query_target(tg, self.robots, t)
                self.events.append({"t": t, "event": "query", "query_id": res.query_id, "target_id": tg.id,
                                    "robot_id": res.robot_id, "reason": res.reason})

    def _log_step(self) -> None:
        n_msgs = self.comm.messages_sent()
        for tg in self.targets:
            for r in self.robots:
                b = r.beliefs.get(tg.id, Belief())
                err = float(np.linalg.norm(b.mean - tg.position)) if b.valid else np.nan
                self.step_log.append({
                    "t": self.t, "target_id": tg.id, "robot_id": r.id,
                    "true_x": tg.position[0], "true_y": tg.position[1],
                    "robot_x": r.position[0], "robot_y": r.position[1],
                    "belief_x": b.mean[0] if b.valid else np.nan,
                    "belief_y": b.mean[1] if b.valid else np.nan,
                    "belief_std": float(np.sqrt(np.trace(b.cov) / 2)) if b.cov is not None else np.nan,
                    "error": err,
                    "observed": r.last_observations.get(tg.id) is not None,
                    "robot_failed": r.failed, "robot_mode": r.mode,
                    "messages_sent_cum": n_msgs,
                })

    # ---- outputs -----------------------------------------------------------
    def step_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.step_log)

    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def dispatch_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.dispatcher.results])

    def summary(self) -> Dict[str, Any]:
        """Per-trial scalar metrics."""
        df = self.step_dataframe()
        out: Dict[str, Any] = {"seed": self.seed, "strategy": self.cfg.strategy, "n_robots": self.cfg.n_robots,
                               "packet_loss_prob": self.cfg.packet_loss_prob,
                               "n_robot_failures": len(self.cfg.robot_failures)}
        if df.empty:
            return out
        # best (min-error) belief across robots at each time -> team error
        team = df.groupby(["t", "target_id"])["error"].min()
        valid = team.dropna()
        out["rmse_team"] = float(np.sqrt(np.nanmean(valid ** 2))) if len(valid) else np.nan
        out["mean_error_team"] = float(valid.mean()) if len(valid) else np.nan
        out["rmse_all_robots"] = float(np.sqrt(np.nanmean(df["error"] ** 2))) if df["error"].notna().any() else np.nan
        out["frac_time_localised"] = float((team <= self.cfg.localised_threshold).mean())
        first = team[team <= self.cfg.localised_threshold]
        out["time_to_first_localisation"] = float(first.index[0][0]) if len(first) else np.nan
        out["final_error_team"] = float(team.iloc[-1]) if len(team) else np.nan
        out["messages_sent"] = self.comm.messages_sent()
        out["messages_delivered"] = self.comm.messages_delivered()
        out["bytes_sent"] = self.comm.bytes_sent()
        out["total_distance"] = float(sum(r.distance_travelled for r in self.robots))
        out["n_relocations"] = int(sum(len(tg.history) for tg in self.targets))
        disp = [r for r in self.dispatcher.results if r.success is not None]
        out["n_queries"] = len(disp)
        out["dispatch_success_rate"] = float(np.mean([r.success for r in disp])) if disp else np.nan
        ok = [r.duration for r in disp if r.success]
        out["dispatch_time_mean"] = float(np.mean(ok)) if ok else np.nan
        return out


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------
def _ci95(x: pd.Series) -> float:
    x = x.dropna()
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def aggregate(trials: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """mean / std / 95% CI per metric, optionally grouped by config columns."""
    metric_cols = [c for c in trials.columns if trials[c].dtype.kind in "fi" and c not in ("seed",)]
    if group_cols:
        g = trials.groupby(group_cols)[metric_cols]
        out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1), "ci95": g.agg(_ci95), "n": g.count()}, axis=1)
        out.columns = [f"{m}_{s}" for s, m in out.columns]
        return out.reset_index()
    rows = {m: {"mean": trials[m].mean(), "std": trials[m].std(ddof=1), "ci95": _ci95(trials[m]),
                "n": trials[m].count()} for m in metric_cols}
    return pd.DataFrame(rows).T


def run_trials(cfg: SimConfig, n_trials: Optional[int] = None, save: bool = True, verbose: bool = True,
               save_step_logs: bool = True) -> pd.DataFrame:
    """Run ``n_trials`` with seeds ``cfg.seed + i``. Writes per-trial CSVs and
    an aggregated summary under ``<output_dir>/<name>/``. Returns the
    per-trial summary DataFrame."""
    n = cfg.n_trials if n_trials is None else n_trials
    env = load_environment(cfg)
    out_dir = os.path.join(cfg.output_dir, cfg.name)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    rows = []
    t0 = time.time()
    for i in range(n):
        sim = Simulation(cfg, seed=cfg.seed + i, env=env)
        sim.run()
        s = sim.summary()
        s["trial"] = i
        rows.append(s)
        if save and save_step_logs:
            sim.step_dataframe().to_csv(os.path.join(out_dir, f"trial_{i:03d}_steps.csv"), index=False)
            sim.events_dataframe().to_csv(os.path.join(out_dir, f"trial_{i:03d}_events.csv"), index=False)
            sim.dispatch_dataframe().to_csv(os.path.join(out_dir, f"trial_{i:03d}_dispatch.csv"), index=False)
        if verbose and (i % max(1, n // 10) == 0 or i == n - 1):
            print(f"[{cfg.name}] trial {i + 1}/{n}  rmse_team={s.get('rmse_team', np.nan):.3f}  "
                  f"localised={s.get('frac_time_localised', np.nan):.2f}  ({time.time() - t0:.1f}s)")
    trials = pd.DataFrame(rows)
    if save:
        trials.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
        aggregate(trials).to_csv(os.path.join(out_dir, "summary.csv"))
    return trials


def run_sweep(base: SimConfig, param: str, values: List[Any], n_trials: Optional[int] = None,
              save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run ``base`` once per value of ``param`` (e.g. n_robots, strategy,
    packet_loss_prob). Returns the concatenated per-trial summaries with a
    ``param``/``value`` column pair."""
    frames = []
    for v in values:
        cfg = base.with_overrides(**{param: v}, name=f"{base.name}_{param}={v}")
        df = run_trials(cfg, n_trials=n_trials, save=save, verbose=verbose, save_step_logs=False)
        df["sweep_param"], df["sweep_value"] = param, v
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    if save:
        os.makedirs(os.path.join(base.output_dir, base.name), exist_ok=True)
        allf.to_csv(os.path.join(base.output_dir, base.name, f"sweep_{param}.csv"), index=False)
    return allf
