"""Minimal communication-free Explore -> Track -> Reacquire navigation.

Only the robot's own observation and Kalman belief may select a target goal.
Patrol, A* planning, obstacle clearance and local robot avoidance are reused.
No true target position or other robot state is passed into this controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .environment import Environment
from .planning import plan_path
from .robot import Belief, Robot


@dataclass
class FollowState:
    mode: str = "explore"  # explore | track | reacquire
    last_goal: np.ndarray | None = None
    last_plan_t: float = -np.inf
    reacquire_start_t: float = -np.inf


class LocalTargetFollower:
    """One follower per robot, entirely private to the robot."""

    def __init__(self, env: Environment, *, stop_radius: float = 0.8,
                 goal_change_distance: float = 0.5, replan_interval: float = 1.0,
                 reacquire_timeout: float = 20.0):
        if stop_radius <= 0 or goal_change_distance <= 0 or replan_interval <= 0 or reacquire_timeout <= 0:
            raise ValueError("tracking parameters must be positive")
        self.env = env
        self.stop_radius = stop_radius
        self.goal_change_distance = goal_change_distance
        self.replan_interval = replan_interval
        self.reacquire_timeout = reacquire_timeout
        self.state = FollowState()

    @property
    def active(self) -> bool:
        return self.state.mode != "explore"

    def reset(self) -> None:
        self.state = FollowState()

    def _start_path(self, robot: Robot, goal: np.ndarray, t: float) -> bool:
        """Follow local belief only. Do not accept an unreachable goal."""
        path = plan_path(robot.position, goal, self.env)
        if not path:
            return False
        robot.set_path(path)
        self.state.last_goal = np.asarray(goal, dtype=float).copy()
        self.state.last_plan_t = t
        return True

    def _resume_exploration(self, robot: Robot) -> None:
        robot.set_path([])
        self.reset()

    def step(self, robot: Robot, belief: Belief, observation: np.ndarray | None,
             t: float) -> str:
        """Update navigation intent; returns current exploration/tracking state.

        For simplicity this MVP follows ONE target (target id 0, chosen by the
        harness). Observations come from Robot.sense, and ``belief`` must be
        the robot's own estimate after the Kalman update.
        """
        if robot.failed:
            self.reset()
            return self.state.mode
        if robot.mode != "patrol":
            # Never override explicit dispatch or another navigation owner.
            self.reset()
            return self.state.mode

        if observation is not None and belief.valid:
            goal = np.asarray(belief.mean, dtype=float)
            was_tracking = self.state.mode == "track"
            self.state.mode = "track"
            self.state.reacquire_start_t = -np.inf
            if np.linalg.norm(robot.position - goal) <= self.stop_radius:
                # Stay close enough to observe; don't jitter around noisy samples.
                robot.set_path([])
                self.state.last_goal = goal.copy()
                return self.state.mode
            should_replan = (not was_tracking or not robot.has_path or
                             self.state.last_goal is None or
                             (np.linalg.norm(goal - self.state.last_goal) >= self.goal_change_distance and
                              t - self.state.last_plan_t >= self.replan_interval))
            if should_replan and not self._start_path(robot, goal, t):
                # A noisy estimate can lie inside a wall: don't teleport or
                # navigate toward a fabricated ground-truth target position.
                self._resume_exploration(robot)
            return self.state.mode

        if self.state.mode == "track":
            # The target has disappeared: inspect its last estimated location
            # once; no future target position is known.
            self.state.mode = "reacquire"
            self.state.reacquire_start_t = t
            if self.state.last_goal is None or not belief.valid:
                self._resume_exploration(robot)
            elif np.linalg.norm(robot.position - self.state.last_goal) <= self.stop_radius:
                self._resume_exploration(robot)
            elif not self._start_path(robot, self.state.last_goal, t):
                self._resume_exploration(robot)

        elif self.state.mode == "reacquire":
            if (t - self.state.reacquire_start_t >= self.reacquire_timeout or
                self.state.last_goal is None or
                np.linalg.norm(robot.position - self.state.last_goal) <= self.stop_radius or
                not robot.has_path):
                self._resume_exploration(robot)

        return self.state.mode
