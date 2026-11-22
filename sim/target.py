"""Targets: true position plus random relocation.

Two motion models:

* ``"carried"`` (default): a person picks the target up and walks it to a new
  random place along a collision-free path at ``carry_speed`` m/s, then puts it
  down and it rests until the next move.  Robots can see it in transit.
* ``"teleport"``: the target jumps to the new place instantly (the original
  model; equivalent to an infinite carry speed).

The rest time between moves is drawn from ``interval_dist``; it is measured
from the moment the target is put down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .environment import Environment
from .planning import plan_path

MOTIONS = ("carried", "teleport")


@dataclass
class RelocationEvent:
    t: float                      # time the move started
    target_id: int
    old_pos: np.ndarray
    new_pos: np.ndarray           # destination
    t_arrive: float = np.nan      # time it was put down (== t for a teleport)
    path: List[np.ndarray] = field(default_factory=list)  # waypoints walked (carried only)


@dataclass
class Target:
    """A target that is moved to a new random free-space position at random
    intervals.

    interval_dist : "uniform" -> rest ~ U(interval_min, interval_max)
                    "exponential" -> rest ~ Exp(mean=interval_mean) (Poisson process)
                    "fixed" -> rest = interval_mean
                    "none" -> never relocates
    motion        : "carried" | "teleport"
    carry_speed   : walking speed of the carrier, m/s (carried only)
    """

    id: int
    position: np.ndarray
    interval_dist: str = "uniform"
    interval_min: float = 20.0
    interval_max: float = 60.0
    interval_mean: float = 40.0
    motion: str = "carried"
    carry_speed: float = 1.2
    min_move_dist: float = 2.0
    next_relocation_time: float = field(default=np.inf)
    paused: bool = False
    history: List[RelocationEvent] = field(default_factory=list)
    # transit state (carried only)
    moving: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _path: List[np.ndarray] = field(default_factory=list)
    _last_t: Optional[float] = None

    def __post_init__(self) -> None:
        if self.motion not in MOTIONS:
            raise ValueError(f"unknown target motion {self.motion!r}; expected one of {MOTIONS}")
        if self.motion == "carried" and not self.carry_speed > 0:
            raise ValueError("carry_speed must be positive")
        self.position = np.asarray(self.position, dtype=float)

    def schedule_next(self, t: float, rng: np.random.Generator) -> None:
        if self.interval_dist == "uniform":
            dt = rng.uniform(self.interval_min, self.interval_max)
        elif self.interval_dist == "exponential":
            dt = rng.exponential(self.interval_mean)
        elif self.interval_dist == "fixed":
            dt = self.interval_mean
        elif self.interval_dist == "none":
            dt = np.inf
        else:
            raise ValueError(f"unknown interval_dist {self.interval_dist!r}")
        self.next_relocation_time = t + dt

    # Hooks used by the dispatch pipeline ---------------------------------
    # Pausing only delays the *start* of the next move; a target already being
    # carried keeps going (the person does not stop because a robot was sent).
    def pause_relocation(self) -> None:
        self.paused = True

    def resume_relocation(self) -> None:
        self.paused = False

    # --------------------------------------------------------------------
    def step(self, t: float, env: Environment, rng: np.random.Generator) -> Optional[RelocationEvent]:
        """Advance to time ``t``.  Returns a RelocationEvent when a move
        *starts* (or, for a teleport, happens); check ``history[-1].t_arrive``
        or the ``moving`` flag for arrivals."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        if self.moving:
            self._advance(dt * self.carry_speed, t, dt, rng)
            return None

        if t < self.next_relocation_time:
            return None
        if self.paused:
            self.next_relocation_time = t + 1e-9  # retry next step
            return None

        old = self.position.copy()
        dest = env.random_free_position(rng, min_dist_from=[old], min_dist=self.min_move_dist)
        path = plan_path(old, dest, env) if self.motion == "carried" else []
        if self.motion == "teleport" or not path:
            # teleport (or, defensively, no walkable path): jump at once
            self.position = dest.copy()
            ev = RelocationEvent(t=t, target_id=self.id, old_pos=old, new_pos=dest.copy(), t_arrive=t)
            self.history.append(ev)
            self.schedule_next(t, rng)
            return ev

        dest = path[-1].copy()  # plan_path may snap the goal to the nearest free cell
        ev = RelocationEvent(t=t, target_id=self.id, old_pos=old, new_pos=dest,
                             path=[p.copy() for p in path])
        self.history.append(ev)
        self._path = [p.copy() for p in path]
        self.moving = True
        self.next_relocation_time = np.inf  # rest is scheduled on arrival
        return ev

    def _advance(self, budget: float, t: float, dt: float, rng: np.random.Generator) -> None:
        start = self.position.copy()
        while budget > 1e-12 and self._path:
            seg = self._path[0] - self.position
            d = float(np.linalg.norm(seg))
            if d <= budget:
                self.position = self._path.pop(0).copy()
                budget -= d
            else:
                self.position = self.position + seg / d * budget
                budget = 0.0
        self.velocity = (self.position - start) / dt if dt > 0 else np.zeros(2)
        if not self._path:
            self.moving = False
            self.velocity = np.zeros(2)
            self.history[-1].t_arrive = t
            self.schedule_next(t, rng)

    @property
    def remaining_path(self) -> List[np.ndarray]:
        return list(self._path)
