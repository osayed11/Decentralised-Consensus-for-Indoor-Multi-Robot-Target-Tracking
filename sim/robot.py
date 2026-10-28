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
    sensor_range: float = 3.0       # m
    comm_range: float = 6.0         # m
    sensor_noise_std: float = 0.3   # m
    waypoint_tol: float = 0.15      # m
    kinematics: str = "holonomic"   # or "diffdrive"
    max_turn_rate: float = 3.0      # rad/s, diffdrive only

    path: List[np.ndarray] = field(default_factory=list)
    failed: bool = False
    failure_time: float = np.inf
    distance_travelled: float = 0.0
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
        """Move toward the next waypoint. Holonomic: straight-line motion at
        up to max_speed. Diff-drive: rotate-then-drive with a turn-rate limit."""
        if self.failed or not self.path:
            return
        target = self.path[0]
        delta = target - self.position
        dist = float(np.linalg.norm(delta))
        if dist < self.waypoint_tol:
            self.path.pop(0)
            return
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        if self.kinematics == "diffdrive":
            err = (desired_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
            max_turn = self.max_turn_rate * dt
            turn = float(np.clip(err, -max_turn, max_turn))
            self.heading += turn
            # slow down while turning sharply
            speed = self.max_speed * max(0.0, np.cos(err))
            step_len = min(speed * dt, dist)
            move = step_len * np.array([np.cos(self.heading), np.sin(self.heading)])
        else:
            self.heading = desired_heading
            step_len = min(self.max_speed * dt, dist)
            move = delta / dist * step_len
        new_pos = self.position + move
        if env.is_free(new_pos):
            self.position = new_pos
            self.distance_travelled += float(np.linalg.norm(move))
        else:
            # Blocked (should be rare with planned paths): drop the path so a
            # new one gets planned.
            self.path = []

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
        self.mode = "idle"
