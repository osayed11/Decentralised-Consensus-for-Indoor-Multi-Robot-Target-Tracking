"""Query / dispatch mechanism.

Infrastructure (complete): ``Dispatcher`` -- pauses relocation, plans a path
to the chosen robot's belief, drives it there, judges success/failure, logs.

############################################################################
#  RESEARCH TEAM CONTRIBUTION -- STUB ONLY: ``select_robot_for_dispatch``. #
#  Placeholder: nearest alive robot by TRUE distance to the target.        #
############################################################################
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .environment import Environment
from .planning import plan_path, path_length
from .robot import Belief, Robot
from .target import Target


def select_robot_for_dispatch(robots: List[Robot], beliefs: Dict[int, Dict[int, Belief]],
                              target: Target, env: Environment) -> Optional[int]:
    """TODO (research team): choose a robot using beliefs / confidence /
    distance-to-belief etc.

    ``beliefs[robot_id][target_id]`` -> Belief.
    Placeholder: nearest alive robot by true distance (cheats; for pipeline
    testing only)."""
    alive = [r for r in robots if not r.failed]
    if not alive:
        return None
    return min(alive, key=lambda r: np.linalg.norm(r.position - target.position)).id


def _brake(robot: Robot, stop_dist: float = 0.35) -> None:
    """Replace the robot's path with a short stop point along its current
    segment so it decelerates smoothly instead of halting instantly."""
    if robot.failed or not robot.path:
        robot.path = []
        return
    wp = robot.path[0]
    seg = wp - robot.position
    d = float(np.linalg.norm(seg))
    robot.path = [wp] if d <= stop_dist else [robot.position + seg / d * stop_dist]


@dataclass
class DispatchResult:
    query_id: int
    target_id: int
    robot_id: Optional[int]
    t_start: float
    t_end: float = np.nan
    success: Optional[bool] = None
    reason: str = ""
    belief_error_at_dispatch: float = np.nan
    planned_path_length: float = np.nan
    distance_travelled: float = np.nan

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class ActiveDispatch:
    result: DispatchResult
    robot: Robot
    target: Target
    start_distance: float
    goal: np.ndarray


@dataclass
class Dispatcher:
    env: Environment
    success_radius: float = 1.0      # robot within this of TRUE target => success
    timeout: float = 120.0           # seconds per dispatch
    replan_on_belief_change: bool = True
    selector: Callable = select_robot_for_dispatch
    results: List[DispatchResult] = field(default_factory=list)
    active: Dict[int, ActiveDispatch] = field(default_factory=dict)  # target_id -> dispatch
    _next_query_id: int = 0

    def query_target(self, target: Target, robots: List[Robot], t: float) -> DispatchResult:
        """Simulates "user asks where the item is"."""
        qid = self._next_query_id
        self._next_query_id += 1
        res = DispatchResult(query_id=qid, target_id=target.id, robot_id=None, t_start=t)
        if target.id in self.active:
            res.t_end, res.success, res.reason = t, False, "already_dispatching"
            self.results.append(res)
            return res
        beliefs = {r.id: r.beliefs for r in robots}
        rid = self.selector(robots, beliefs, target, self.env)
        if rid is None:
            res.t_end, res.success, res.reason = t, False, "no_robot_available"
            self.results.append(res)
            return res
        robot = next(r for r in robots if r.id == rid)
        belief = robot.beliefs.get(target.id, Belief())
        if not belief.valid:
            res.robot_id = rid
            res.t_end, res.success, res.reason = t, False, "no_belief"
            self.results.append(res)
            return res

        target.pause_relocation()
        goal = np.asarray(belief.mean, dtype=float)
        path = plan_path(robot.position, goal, self.env)
        res.robot_id = rid
        res.belief_error_at_dispatch = float(np.linalg.norm(goal - target.position))
        res.planned_path_length = path_length([robot.position] + path) if path else np.nan
        if not path and np.linalg.norm(robot.position - goal) > robot.waypoint_tol:
            target.resume_relocation()
            res.t_end, res.success, res.reason = t, False, "no_path"
            self.results.append(res)
            return res
        robot.set_path(path)
        robot.mode = "dispatch"
        self.active[target.id] = ActiveDispatch(res, robot, target, robot.distance_travelled, goal)
        self.results.append(res)
        return res

    def step(self, t: float, robots: List[Robot]) -> List[DispatchResult]:
        """Advance active dispatches; returns the ones finished this step."""
        finished = []
        for tid, ad in list(self.active.items()):
            r, tg, res = ad.robot, ad.target, ad.result
            dist_true = float(np.linalg.norm(r.position - tg.position))
            if dist_true <= self.success_radius:
                self._finish(ad, t, True, "reached_target")
            elif r.failed:
                self._finish(ad, t, False, "robot_failed")
            elif t - res.t_start > self.timeout:
                self._finish(ad, t, False, "timeout")
            elif not r.has_path:
                # Arrived at the belief position. Re-plan if the belief moved
                # (e.g. the strategy updated it en route), else fail.
                b = r.beliefs.get(tid, Belief())
                if self.replan_on_belief_change and b.valid and np.linalg.norm(b.mean - ad.goal) > r.waypoint_tol:
                    ad.goal = np.asarray(b.mean, dtype=float)
                    r.set_path(plan_path(r.position, ad.goal, self.env))
                    if not r.has_path:
                        self._finish(ad, t, False, "no_path_on_replan")
                else:
                    self._finish(ad, t, False, "target_not_at_belief")
            else:
                continue
            finished.append(res)
        return finished

    def _finish(self, ad: ActiveDispatch, t: float, success: bool, reason: str) -> None:
        res = ad.result
        res.t_end, res.success, res.reason = t, success, reason
        res.distance_travelled = ad.robot.distance_travelled - ad.start_distance
        ad.robot.mode = "patrol" if not ad.robot.failed else "idle"
        _brake(ad.robot)
        ad.target.resume_relocation()
        del self.active[ad.target.id]
