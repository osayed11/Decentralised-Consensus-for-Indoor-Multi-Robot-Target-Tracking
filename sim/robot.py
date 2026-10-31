"""Robot model: holonomic point-mass kinematics, waypoint following and the
simulated range/noise sensor.  Belief storage lives here but belief *update*
logic is in strategies.py (research team)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .environment import Environment
from .target import Target


@dataclass
class Belief:
    """Position estimate for one target. ``cov`` may be None if the strategy
    does not track uncertainty."""

    mean: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    timestamp: float = -np.inf
    source: str = "none"  # free-form tag, e.g. "own_obs", "fused", "msg"

    @property
    def valid(self) -> bool:
        return self.mean is not None


@dataclass
class Robot:
    id: int
    position: np.ndarray
    heading: float = 0.0
    max_speed: float = 1.0          # m/s
    max_accel: float = 1.0          # m/s^2 (velocity ramping; 0 or inf = instant)
    sensor_range: float = 3.0       # m
    comm_range: float = 6.0         # m
    sensor_noise_std: float = 0.3   # m
    waypoint_tol: float = 0.15      # m, final waypoint
    corner_tol: float = 0.4         # m, intermediate waypoints (lookahead -> rounded corners)
    kinematics: str = "holonomic"   # or "diffdrive"
    max_turn_rate: float = 3.0      # rad/s, diffdrive only

    path: List[np.ndarray] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    failed: bool = False
    failure_time: float = np.inf
    distance_travelled: float = 0.0
    blocked_count: int = 0
    beliefs: Dict[int, Belief] = field(default_factory=dict)
    last_observations: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    inbox: List[dict] = field(default_factory=list)
    mode: str = "patrol"            # "patrol" | "dispatch" | "idle"

    # ---- motion --------------------------------------------------------
    def set_path(self, waypoints: List[np.ndarray]) -> None:
        self.path = [np.asarray(w, dtype=float) for w in waypoints]

    @property
    def has_path(self) -> bool:
        return len(self.path) > 0

    def step(self, dt: float, env: Environment) -> None:
        """Move toward the next waypoint.

        Holonomic: velocity ramps toward the waypoint direction at up to
        ``max_accel`` and decelerates into the final waypoint, which gives
        smooth starts/stops and rounded corners.  Diff-drive: rotate-then-
        drive with a turn-rate limit."""
        if self.failed or not self.path:
            self.velocity = np.zeros(2)
            return
        # Pop waypoints we are close enough to.  Intermediate waypoints use a
        # looser tolerance (rounded corners) but only if the shortcut to the
        # following waypoint is clear on the inflated grid.
        while self.path:
            d0 = np.linalg.norm(self.path[0] - self.position)
            if len(self.path) == 1:
                if d0 < self.waypoint_tol:
                    self.path.pop(0)
                else:
                    break
            elif d0 < 1e-6 or (d0 < self.corner_tol and env.line_of_sight(self.position, self.path[1], inflated=True)):
                # Intermediate waypoint: only skip it if the shortcut to the
                # next one is verified clear (the motion loop below passes
                # exactly through waypoints otherwise).
                self.path.pop(0)
            else:
                break
        if not self.path:
            self.velocity = np.zeros(2)
            return
        target = self.path[0]
        delta = target - self.position
        dist = float(np.linalg.norm(delta))
        desired_heading = float(np.arctan2(delta[1], delta[0]))

        if self.kinematics == "diffdrive":
            err = (desired_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
            max_turn = self.max_turn_rate * dt
            self.heading += float(np.clip(err, -max_turn, max_turn))
            speed = self.max_speed * max(0.0, np.cos(err))
            step_len = min(speed * dt, dist)
            move = step_len * np.array([np.cos(self.heading), np.sin(self.heading)])
            self.velocity = move / dt
        else:
            # Speed ramps (tangential acceleration limit); direction follows
            # the planned segment exactly so the LOS-verified corridor is kept.
            remaining = dist + sum(float(np.linalg.norm(self.path[i + 1] - self.path[i]))
                                   for i in range(len(self.path) - 1))
            speed = float(np.linalg.norm(self.velocity))
            if self.max_accel and np.isfinite(self.max_accel) and self.max_accel > 0:
                brake_dist = max(remaining - self.waypoint_tol, 0.0)
                target_speed = min(self.max_speed, np.sqrt(2 * self.max_accel * brake_dist) + 0.05)
                # accelerate at most max_accel; braking tracks the profile
                # exactly (the sqrt profile *is* constant deceleration).
                speed = min(speed + self.max_accel * dt, target_speed)
            else:
                speed = self.max_speed
            # Advance ``speed * dt`` along the polyline, carrying leftover
            # distance past intermediate waypoints so corners don't stall.
            budget = speed * dt
            pos = self.position.copy()
            while self.path and budget > 0:
                seg = self.path[0] - pos
                seg_len = float(np.linalg.norm(seg))
                if seg_len <= budget and len(self.path) > 1:
                    pos = self.path[0].copy()
                    budget -= seg_len
                    self.path.pop(0)
                else:
                    pos = pos + seg / max(seg_len, 1e-9) * min(budget, seg_len)
                    budget = 0.0
            move = pos - self.position
            if np.linalg.norm(move) > 1e-9:
                self.heading = float(np.arctan2(move[1], move[0]))
            self.velocity = move / dt

        new_pos = self.position + move
        if env.is_free(new_pos):
            self.position = new_pos
            self.distance_travelled += float(np.linalg.norm(move))
        else:
            # Blocked (rare with planned paths): stop and drop the path so a
            # new one gets planned.
            self.blocked_count += 1
            self.path = []
            self.velocity = np.zeros(2)

    # ---- sensing -------------------------------------------------------
    def sense(self, targets: List[Target], env: Environment, rng: np.random.Generator,
              use_line_of_sight: bool = False) -> Dict[int, Optional[np.ndarray]]:
        """Per target: noisy position if within sensor_range (and optionally
        line-of-sight), else None."""
        obs: Dict[int, Optional[np.ndarray]] = {}
        for tg in targets:
            if self.failed:
                obs[tg.id] = None
                continue
            d = float(np.linalg.norm(tg.position - self.position))
            if d <= self.sensor_range and (not use_line_of_sight or env.line_of_sight(self.position, tg.position)):
                obs[tg.id] = tg.position + rng.normal(0.0, self.sensor_noise_std, size=2)
            else:
                obs[tg.id] = None
        self.last_observations = obs
        return obs

    # ---- faults --------------------------------------------------------
    def fail(self, t: float) -> None:
        self.failed = True
        self.failure_time = t
        self.path = []
        self.velocity = np.zeros(2)
        self.mode = "idle"
