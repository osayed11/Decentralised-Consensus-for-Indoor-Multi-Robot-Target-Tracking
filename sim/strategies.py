"""Coordination strategies.

Each timestep the simulation, for every alive robot, calls:
   1. msgs = robot.inbox (delivered since last step; cleared after)
   2. for each target: belief = strategy.update_belief(robot, target_id,
          observation, msgs, t)
   3. strategy.detect_relocation(robot, target_id, observation, belief)
   4. strategy.get_messages_to_send(robot, comm_layer, robots, t)
      -> list of (receiver_id | None for broadcast, payload dict)

All three strategies share the per-robot estimator in ``sim/estimator.py``
(Kalman filter, innovation gating, belief reset) and the message formats in
``sim/messages.py``.  What is still TODO is marked per class below.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .comms import COORDINATOR, CommunicationLayer
from .estimator import Estimator, EstimatorConfig, chi2_threshold
from .messages import (ABSENCE, OBSERVATION, absence_message, belief_message,
                       observation_message, for_target, message_to_belief,
                       contradicted_by_absence)
from .robot import Belief, Robot

Message = Tuple[Optional[int], Dict[str, Any]]  # (receiver_id, None=broadcast, or COORDINATOR; payload)


class CoordinationStrategy:
    """Interface. Subclass and override the three methods below."""

    name = "base"

    def reset(self, robots: List[Robot], rng: np.random.Generator) -> None:
        """Called once at the start of each trial."""
        self.rng = rng

    def bind(self, env, use_line_of_sight: bool = False) -> None:
        """Called by the simulation after ``reset``: the map and sensing mode,
        needed to decide whether a believed position is in view."""
        self.env = env
        self.use_line_of_sight = use_line_of_sight

    def detect_absence(self, robot: Robot, target_id: int) -> bool:
        """True if this step's update found the believed position empty
        (no-sighting rule).  Default: never."""
        return False

    def update_belief(self, robot: Robot, target_id: int, observation: Optional[np.ndarray],
                      neighbor_messages: List[dict], t: float) -> Belief:
        """Returns updated belief (position estimate + confidence/covariance).
        TO BE IMPLEMENTED: independent / centralized / decentralized logic."""
        raise NotImplementedError

    def detect_relocation(self, robot: Robot, target_id: int, observation: Optional[np.ndarray],
                          belief: Belief) -> bool:
        """Innovation gating / relocation detection.
        TO BE IMPLEMENTED. Return True if the target appears to have moved."""
        raise NotImplementedError

    def get_messages_to_send(self, robot: Robot, comm: CommunicationLayer, robots: List[Robot],
                             t: float) -> List[Message]:
        """Returns messages this robot sends this timestep.
        TO BE IMPLEMENTED."""
        raise NotImplementedError

    def coordinator_step(self, inbox: List[dict], comm: CommunicationLayer, robots: List[Robot],
                         t: float) -> List[dict]:
        """Runs on the off-board coordinator once per step while it is up, after
        the robots have sent.  ``inbox`` holds what reached it this step.
        Returns payloads to broadcast to every alive robot (they arrive next
        step).  Only the centralised strategy uses it; default: do nothing."""
        return []


# ---------------------------------------------------------------------------
# Shared base: every strategy runs the same per-robot estimator (sim/estimator.py)
# on its own observations.  The strategies differ only in what they send and how
# they fuse what they receive -- the parts still marked TODO below.
# ---------------------------------------------------------------------------
class _SharedEstimatorBase(CoordinationStrategy):
    """Per-robot Kalman filter with innovation gating and belief reset.

    ``strategy_kwargs`` in the config are passed here, e.g.
    ``{gate_prob: 0.99, process_noise: 0.01, confirm: 1}`` (``gate`` sets the
    chi-square threshold directly and overrides ``gate_prob``), plus the
    no-sighting rule: ``absence`` (on/off), ``absence_view_fraction``; and the
    target model: ``model`` ("imm" resting/carried, default, or "static") with
    ``mean_rest``, ``mean_walk``, ``accel_noise``, ``init_speed_std``,
    ``max_pos_std`` (see ``EstimatorConfig``)."""

    def __init__(self, gate: Optional[float] = None, gate_prob: float = 0.99,
                 process_noise: float = 0.01, confirm: int = 1, absence: bool = True,
                 absence_view_fraction: float = 0.8, **model_kwargs):
        self.config = EstimatorConfig(process_noise=process_noise,
                                      gate=gate if gate is not None else chi2_threshold(gate_prob),
                                      confirm=confirm, absence=absence,
                                      absence_view_fraction=absence_view_fraction, **model_kwargs)

    def reset(self, robots, rng):
        super().reset(robots, rng)
        self.estimator = Estimator(self.config)

    def belief_in_view(self, robot: Robot, belief: Belief) -> bool:
        """Was the believed position comfortably in view this step?  Within
        ``absence_view_fraction`` of the sensor range and, if line-of-sight
        sensing is on, not behind a wall (a wall is not an absence)."""
        if not belief.valid:
            return False
        if np.linalg.norm(belief.mean - robot.position) > self.config.absence_view_fraction * robot.sensor_range:
            return False
        env = getattr(self, "env", None)
        if getattr(self, "use_line_of_sight", False) and env is not None:
            return env.line_of_sight(robot.position, belief.mean)
        return True

    def update_belief(self, robot, target_id, observation, neighbor_messages, t):
        prev = robot.beliefs.get(target_id, Belief())
        in_view = observation is None and self.belief_in_view(robot, prev)
        return self.estimator.update(robot.id, target_id, prev, observation, robot.sensor_noise_std, t,
                                     in_view=in_view, view_pos=robot.position,
                                     view_radius=self.config.absence_view_fraction * robot.sensor_range)

    def detect_relocation(self, robot, target_id, observation, belief):
        return self.estimator.relocation_detected(robot.id, target_id)

    def detect_absence(self, robot, target_id):
        return self.estimator.absence_detected(robot.id, target_id)

    def get_messages_to_send(self, robot, comm, robots, t):
        return []


class IndependentStrategy(_SharedEstimatorBase):
    """Each robot tracks on its own; no communication.  Complete as it is."""
    name = "independent"


class CentralizedStrategy(_SharedEstimatorBase):
    """Off-board fusion via COORDINATOR, with independent local fallback.

    Raw observations are fused once at the server. Clients adopt fresh server
    posteriors without feeding correlated evidence into their local filters.
    Absence reports invalidate matching stale beliefs in both directions.
    """
    name = "centralized"

    def reset(self, robots, rng):
        super().reset(robots, rng)
        self._robot_ids = {r.id for r in robots}
        self._local = {}
        self._hub = {}
        self._received = {}
        self._processed = {}
        self._detected = {}
        self._absent = {}

    def _empty_area_contradicts(self, pos, last_seen, checked_pos, checked_at, radius):
        if not contradicted_by_absence(pos, last_seen, checked_pos, checked_at, radius):
            return False
        env = getattr(self, 'env', None)
        return not (getattr(self, 'use_line_of_sight', False) and env is not None
                    and not env.line_of_sight(checked_pos, pos))

    def _belief_contradicted(self, pos, last_seen, belief):
        return (belief.absence_pos is not None and self._empty_area_contradicts(
            pos, last_seen, belief.absence_pos, belief.last_absence, belief.absence_radius))

    def _receive_absence(self, belief, msg):
        if belief.valid and self._empty_area_contradicts(
                belief.mean, belief.last_seen, msg['pos'], msg['t_obs'], msg['radius']):
            return Estimator.apply_absence(belief, msg['t_obs'], msg['pos'], msg['radius'])
        if msg['t_obs'] > belief.last_absence:
            return replace(belief, last_absence=msg['t_obs'],
                           absence_pos=np.array(msg['pos'], dtype=float), absence_radius=msg['radius'])
        return belief

    def update_belief(self, robot, target_id, observation, neighbor_messages, t):
        key = (robot.id, target_id)
        # Recovery may explicitly clear public beliefs; do not resurrect hidden state.
        if target_id not in robot.beliefs and key in self._local:
            self._local.pop(key, None)
            self._received.pop(key, None)
        prev = self._local.get(key, Belief())
        local = self.estimator.update(
            robot.id, target_id, prev, observation, robot.sensor_noise_std, t,
            in_view=observation is None and self.belief_in_view(robot, prev),
            view_pos=robot.position,
            view_radius=self.config.absence_view_fraction * robot.sensor_range)
        self._detected[key] = self.estimator.relocation_detected(*key)
        self._absent[key] = self.estimator.absence_detected(*key)
        cached = self._received.get(key, Belief())
        for msg in for_target(neighbor_messages, target_id, ABSENCE):
            if self.config.absence and msg['sender'] == COORDINATOR and msg['t_obs'] <= msg['t_sent'] <= t:
                local = self._receive_absence(local, msg)
                cached = self._receive_absence(cached, msg)
        for msg in sorted(for_target(neighbor_messages, target_id), key=lambda m: m['t_sent']):
            if (msg['sender'] == COORDINATOR and msg['last_seen'] <= msg['t_sent'] <= t
                    and msg['t_sent'] > cached.timestamp
                    and msg['last_seen'] >= cached.last_seen
                    and not self._belief_contradicted(msg['mean'], msg['last_seen'], local)
                    and not self._belief_contradicted(msg['mean'], msg['last_seen'], cached)):
                cached = message_to_belief(msg)
        if cached.valid and self._belief_contradicted(cached.mean, cached.last_seen, local):
            cached = self.estimator.apply_absence(cached, local.last_absence,
                                                  local.absence_pos, local.absence_radius)
        selected = cached if cached.valid and (not local.valid or cached.last_seen >= local.last_seen) else local
        # A robot can inspect a server-provided position even without its own prior sighting.
        if (self.config.absence and observation is None and self.belief_in_view(robot, selected)):
            checked = self.estimator.update(
                robot.id, target_id, selected, None, robot.sensor_noise_std, t,
                in_view=True, view_pos=robot.position,
                view_radius=self.config.absence_view_fraction * robot.sensor_range)
            msg = absence_message(robot.id, target_id, checked, t)
            local = self._receive_absence(local, msg)
            cached = self._receive_absence(cached, msg)
            selected = checked
            self._absent[key] = True
        self._local[key], self._received[key] = local, cached
        return replace(self.estimator.predict(selected, t), reset_count=local.reset_count)

    def detect_relocation(self, robot, target_id, observation, belief):
        return self._detected.get((robot.id, target_id), False)

    def detect_absence(self, robot, target_id):
        return self._absent.get((robot.id, target_id), False)

    def get_messages_to_send(self, robot, comm, robots, t):
        if robot.failed:
            return []
        out = [(COORDINATOR, observation_message(robot.id, tid, z, robot.sensor_noise_std, t))
               for tid, z in robot.last_observations.items() if z is not None]
        for tid, belief in robot.beliefs.items():
            if self.config.absence and not belief.valid and belief.absence_pos is not None:
                out.append((COORDINATOR, absence_message(robot.id, tid, belief, t)))
        return out

    def coordinator_step(self, inbox, comm, robots, t):
        if not comm.coordinator_up:
            return []
        accepted = [m for m in inbox if m.get('type') in (OBSERVATION, ABSENCE)
                    and m.get('sender') in self._robot_ids and m['t_obs'] <= m['t_sent'] <= t]
        for msg in sorted(accepted, key=lambda m: (m['t_obs'], m['sender'], m['type'])):
            tid, when = msg['target_id'], msg['t_obs']
            origin = (msg['sender'], tid, msg['type'])
            if when <= self._processed.get(origin, -np.inf):
                continue
            self._processed[origin] = when
            belief = self._hub.get(tid, Belief())
            if msg['type'] == ABSENCE:
                if self.config.absence:
                    belief = self._receive_absence(belief, msg)
            elif when >= belief.timestamp and not self._belief_contradicted(msg['z'], when, belief):
                noise_std = max(msg['noise_std'], np.sqrt(self.config.min_cov))
                belief = self.estimator.update(COORDINATOR, tid, belief, msg['z'], noise_std, when)
            self._hub[tid] = belief
        out = []
        for tid, belief in self._hub.items():
            if belief.valid:
                out.append(belief_message(COORDINATOR, tid, self.estimator.predict(belief, t), t))
            elif belief.absence_pos is not None:
                out.append(absence_message(COORDINATOR, tid, belief, t))
        return out


class DecentralizedConsensusStrategy(_SharedEstimatorBase):
    """Neighbours exchange estimates and run a consensus update.

    Messaging is in place: every robot broadcasts its belief about every target to
    its graph neighbours each step (``messages.belief_message``).
    TODO (research team): the consensus update in ``update_belief`` -- use
    ``messages.for_target(neighbor_messages, target_id)``, ``messages.is_newer``
    and the estimator's gating to decide between fusing and resetting, and
    ``messages.belief_contradicted`` so a neighbour's stale belief about a spot I
    found empty is ignored, and ``messages.for_target(..., messages.ABSENCE)`` so a
    neighbour's "not there" clears my matching stale belief."""
    name = "decentralized"

    def get_messages_to_send(self, robot, comm, robots, t):
        """Broadcast my belief about each target; if the no-sighting rule cleared
        it, broadcast the empty check instead (a robot that knows "not there"
        has nothing else to share)."""
        out = []
        for target_id, belief in robot.beliefs.items():
            if belief.valid and belief.cov is not None:
                out.append((None, belief_message(robot.id, target_id, belief, t)))
            elif belief.absence_pos is not None:
                out.append((None, absence_message(robot.id, target_id, belief, t)))
        return out


STRATEGIES = {
    IndependentStrategy.name: IndependentStrategy,
    CentralizedStrategy.name: CentralizedStrategy,
    DecentralizedConsensusStrategy.name: DecentralizedConsensusStrategy,
}


def make_strategy(name: str, **kwargs) -> CoordinationStrategy:
    try:
        return STRATEGIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {list(STRATEGIES)}")
