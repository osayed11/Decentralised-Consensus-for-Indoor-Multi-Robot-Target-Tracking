"""Fault injection: unexpected robot and coordinator (server) failures.

Failure model (decided 2026-11-22, see SHARED_COMPONENTS.md "Failures"):

* **Unexpected = constant risk.**  Each robot fails during a trial with
  probability ``robot_failure_prob``; its failure time is exponential (constant
  hazard), so nothing before the failure hints that it is coming.  With
  P(fail before T) = p the closed form is  t = T * ln(1 - u) / ln(1 - p),
  u ~ U(0, 1), and the robot fails iff u < p.  At most one failure per robot per
  trial.  The coordinator (central server) has its own ``coordinator_failure_prob``.
* **Fixed-time failures** (``robot_failures: {id: t}``, ``coordinator_failure_time``)
  override the random draw, for controlled scenarios and demos.
* **Downtime**: ``None`` = permanent (default).  A number = the robot / server
  comes back after that many seconds.  A recovering robot keeps its (now stale)
  beliefs unless ``keep_belief_on_recovery`` is False (a reboot that loses memory).
* **Fairness**: every failure time is drawn once, at the start of the trial, from
  the fault model's own random stream.  The same seed therefore gives the same
  failures for every strategy, and switching failures on or off leaves the rest
  of the world (targets, other robots' routes and sensing) unchanged.

A failed robot stops sensing, moving and communicating.  A failed coordinator
neither receives nor broadcasts (see ``CommunicationLayer``); robots then only
have their own estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .robot import Robot

INF = float("inf")


def sample_failure_time(u: float, p: float, horizon: float) -> float:
    """Constant-hazard failure time with P(fail before ``horizon``) = p.  Returns
    inf if this draw does not fail within the horizon."""
    if not 0.0 <= p < 1.0:
        raise ValueError("failure probability must be in [0, 1)")
    if p == 0.0 or u >= p:
        return INF
    return float(horizon * np.log1p(-u) / np.log1p(-p))


@dataclass
class FaultSchedule:
    # robots
    robot_failures: Dict[int, float] = field(default_factory=dict)  # fixed: robot_id -> failure time (s)
    robot_failure_prob: float = 0.0
    robot_downtime: Optional[float] = None          # None = permanent
    keep_belief_on_recovery: bool = True
    # coordinator (central server)
    coordinator_failure_time: Optional[float] = None
    coordinator_failure_prob: float = 0.0
    coordinator_downtime: Optional[float] = None    # None = permanent
    # other
    packet_loss_prob: float = 0.0                   # applied by CommunicationLayer
    horizon: float = 300.0                          # trial length used for the probabilities
    # resolved schedule: id -> (t_fail, t_recover)
    robot_schedule: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    coordinator_schedule: Tuple[float, float] = (INF, INF)
    log: List[dict] = field(default_factory=list)

    def sample(self, robot_ids: List[int], rng: np.random.Generator) -> None:
        """Resolve all failure times up front.  Always draws one number per robot
        plus one for the coordinator, so the stream is aligned whatever the
        settings."""
        down_r = INF if self.robot_downtime is None else float(self.robot_downtime)
        self.robot_schedule = {}
        for rid in robot_ids:
            t_fail = sample_failure_time(rng.random(), self.robot_failure_prob, self.horizon)
            if rid in self.robot_failures:
                t_fail = float(self.robot_failures[rid])
            if np.isfinite(t_fail):
                self.robot_schedule[rid] = (t_fail, t_fail + down_r)
        u = rng.random()
        t_fail = sample_failure_time(u, self.coordinator_failure_prob, self.horizon)
        if self.coordinator_failure_time is not None:
            t_fail = float(self.coordinator_failure_time)
        down_c = INF if self.coordinator_downtime is None else float(self.coordinator_downtime)
        self.coordinator_schedule = (t_fail, t_fail + down_c) if np.isfinite(t_fail) else (INF, INF)

    def apply(self, t: float, robots: List[Robot], comm=None) -> List[dict]:
        """Fail / recover robots and the coordinator at time ``t``.  Returns the
        events that happened this step (also appended to ``self.log``)."""
        events: List[dict] = []
        for r in robots:
            sched = self.robot_schedule.get(r.id)
            if sched is None:
                continue
            t_fail, t_rec = sched
            if not r.failed and t_fail <= t < t_rec:
                r.fail(t)
                events.append({"t": t, "event": "robot_failure", "robot_id": r.id})
            elif r.failed and t >= t_rec:
                r.recover(t, keep_belief=self.keep_belief_on_recovery)
                events.append({"t": t, "event": "robot_recovery", "robot_id": r.id})
        if comm is not None:
            t_fail, t_rec = self.coordinator_schedule
            should_be_up = not (t_fail <= t < t_rec)
            if comm.coordinator_up and not should_be_up:
                comm.set_coordinator_up(False)
                events.append({"t": t, "event": "coordinator_failure"})
            elif not comm.coordinator_up and should_be_up:
                comm.set_coordinator_up(True)
                events.append({"t": t, "event": "coordinator_recovery"})
        self.log.extend(events)
        return events
