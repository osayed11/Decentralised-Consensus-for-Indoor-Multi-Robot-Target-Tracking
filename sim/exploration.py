"""Private exploration memory and robust random-waypoint patrol.

Phase 1: patrol destinations are independent uniform random reachable locations.
Phase 2: each robot remembers when its OWN sensor could see each reachable cell.
No other robot's location, belief, observation, or exploration map is consulted.

The Week 4 lab's discrete grid representation motivates the memory, but this
module does NOT implement Voronoi/Lloyd coverage control (future phase).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Environment
from .planning import plan_path
from .robot import Robot


class ExplorationMemory:
    """Last-observed time per robot-reachable cell; -inf means never observed.

    The denominator is the reachable inflated free-space grid, not obstacle
    cells. A cell is observed when its centre is within sensor range and,
    optionally, there is clear raw-grid line of sight to that centre.
    """

    def __init__(self, env: Environment):
        self.last_seen = np.full(env.grid.shape, -np.inf, dtype=float)
        self.reachable = env.reachable.copy()
        self.resolution = env.resolution
        self._rows, self._cols = np.nonzero(self.reachable)
        self._xy = np.column_stack(((self._cols + 0.5) * env.resolution,
                                    (self._rows + 0.5) * env.resolution))

    def observe(self, position: np.ndarray, sensor_range: float, t: float,
                env: Environment, use_line_of_sight: bool = False) -> int:
        """Mark currently observable cells; return number of marked cells.

        This uses geometry ONLY, never target truth or a detection outcome.
        Sensing a cell doesn't imply a target was actually detected there.
        """
        if sensor_range < 0:
            raise ValueError("sensor_range must be non-negative")
        delta = self._xy - np.asarray(position, dtype=float)
        candidates = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= sensor_range**2 + 1e-12)
        marked = 0
        for ix in candidates:
            xy = self._xy[ix]
            if not use_line_of_sight or env.line_of_sight(position, xy):
                self.last_seen[self._rows[ix], self._cols[ix]] = t
                marked += 1
        return marked

    @property
    def observed_mask(self) -> np.ndarray:
        return np.isfinite(self.last_seen) & self.reachable

    @property
    def coverage_fraction(self) -> float:
        n = int(self.reachable.sum())
        return float(self.observed_mask.sum() / n) if n else 0.0

    def age(self, t: float) -> np.ndarray:
        """Age of observation; never-seen cells remain +inf."""
        result = np.full_like(self.last_seen, np.inf)
        seen = self.observed_mask
        result[seen] = np.maximum(0.0, t - self.last_seen[seen])
        return result

    def clear(self) -> None:
        """Optional reboot reset (called when keep_belief_on_recovery=False)."""
        self.last_seen.fill(-np.inf)


@dataclass
class PatrolState:
    goal: np.ndarray | None = None
    last_progress_pos: np.ndarray | None = None
    last_progress_t: float = 0.0
    last_observation_t: float = -np.inf
    goals_assigned: int = 0
    goals_rejected: int = 0
    stalls: int = 0

    def reset_motion(self) -> None:
        self.goal = None
        self.last_progress_pos = None


class IndependentPatrol:
    """Manage one robot's exploration decisions using its private RNG only."""

    def __init__(self, env: Environment, rng: np.random.Generator,
                 min_goal_distance: float = 1.0, max_goal_attempts: int = 20,
                 stall_timeout: float = 8.0, progress_distance: float = 0.15):
        if min_goal_distance < 0 or max_goal_attempts < 1 or stall_timeout <= 0 or progress_distance <= 0:
            raise ValueError("invalid patrol parameters")
        self.env = env
        self.rng = rng
        self.min_goal_distance = min_goal_distance
        self.max_goal_attempts = max_goal_attempts
        self.stall_timeout = stall_timeout
        self.progress_distance = progress_distance
        self.memory = ExplorationMemory(env)
        self.state = PatrolState()

    def observe_if_due(self, robot: Robot, t: float, interval: float,
                       use_line_of_sight: bool = False) -> None:
        if robot.failed:
            return
        if interval <= 0:
            raise ValueError("exploration observation interval must be positive")
        if t + 1e-9 >= self.state.last_observation_t + interval:
            self.memory.observe(robot.position, robot.sensor_range, t, self.env, use_line_of_sight)
            self.state.last_observation_t = t

    def step(self, robot: Robot, t: float) -> None:
        """Select/reselect patrol goal; does not move robot (Robot.step does)."""
        if robot.failed or robot.mode != "patrol":
            # When dispatched, keep the memory but discard patrol goal state.
            self.state.reset_motion()
            return

        if robot.has_path:
            if self.state.last_progress_pos is None:
                self.state.last_progress_pos = robot.position.copy()
                self.state.last_progress_t = t
            elif np.linalg.norm(robot.position - self.state.last_progress_pos) >= self.progress_distance:
                self.state.last_progress_pos = robot.position.copy()
                self.state.last_progress_t = t
            elif t - self.state.last_progress_t >= self.stall_timeout:
                robot.set_path([])
                self.state.stalls += 1
                self.state.reset_motion()

        if robot.has_path:
            return

        self.state.reset_motion()
        for _ in range(self.max_goal_attempts):
            goal = self.env.random_free_position(self.rng)
            if np.linalg.norm(goal - robot.position) < self.min_goal_distance:
                self.state.goals_rejected += 1
                continue
            path = plan_path(robot.position, goal, self.env)
            if not path:
                self.state.goals_rejected += 1
                continue
            # Explicit path checks protect against unusual map discretisation.
            if np.linalg.norm(path[-1] - robot.position) < self.min_goal_distance:
                self.state.goals_rejected += 1
                continue
            robot.set_path(path)
            self.state.goal = goal.copy()
            self.state.last_progress_pos = robot.position.copy()
            self.state.last_progress_t = t
            self.state.goals_assigned += 1
            return
        # No viable goal on this step: remain idle, retry on next step.
        robot.set_path([])
