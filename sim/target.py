"""Targets: true position plus random relocation (teleport)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .environment import Environment


@dataclass
class RelocationEvent:
    t: float
    target_id: int
    old_pos: np.ndarray
    new_pos: np.ndarray


@dataclass
class Target:
    """A target that teleports to a new random free-space position at random
    intervals.

    interval_dist : "uniform" -> interval ~ U(interval_min, interval_max)
                    "exponential" -> interval ~ Exp(mean=interval_mean) (Poisson process)
                    "fixed" -> interval = interval_mean
                    "none" -> never relocates
    """

    id: int
    position: np.ndarray
    interval_dist: str = "uniform"
    interval_min: float = 20.0
    interval_max: float = 60.0
    interval_mean: float = 40.0
    next_relocation_time: float = field(default=np.inf)
    paused: bool = False
    history: List[RelocationEvent] = field(default_factory=list)

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

    # Hooks used by the research team's dispatch logic -------------------
    def pause_relocation(self) -> None:
        self.paused = True

    def resume_relocation(self) -> None:
        self.paused = False

    # --------------------------------------------------------------------
    def step(self, t: float, env: Environment, rng: np.random.Generator) -> Optional[RelocationEvent]:
        """Advance to time ``t``. Relocate if due and not paused.  While
        paused the deadline is simply pushed back, so relocation happens
        shortly after resumption."""
        if t < self.next_relocation_time:
            return None
        if self.paused:
            self.next_relocation_time = t + 1e-9  # retry next step
            return None
        old = self.position.copy()
        self.position = env.random_free_position(rng, min_dist_from=[old], min_dist=2.0)
        ev = RelocationEvent(t=t, target_id=self.id, old_pos=old, new_pos=self.position.copy())
        self.history.append(ev)
        self.schedule_next(t, rng)
        return ev
