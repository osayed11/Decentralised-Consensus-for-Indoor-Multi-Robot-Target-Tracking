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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .comms import CommunicationLayer
from .estimator import Estimator, EstimatorConfig, chi2_threshold
from .messages import belief_message
from .robot import Belief, Robot

Message = Tuple[Optional[int], Dict[str, Any]]  # (receiver_id or None=broadcast, payload)


class CoordinationStrategy:
    """Interface. Subclass and override the three methods below."""

    name = "base"

    def reset(self, robots: List[Robot], rng: np.random.Generator) -> None:
        """Called once at the start of each trial."""
        self.rng = rng

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


# ---------------------------------------------------------------------------
# Shared base: every strategy runs the same per-robot estimator (sim/estimator.py)
# on its own observations.  The strategies differ only in what they send and how
# they fuse what they receive -- the parts still marked TODO below.
# ---------------------------------------------------------------------------
class _SharedEstimatorBase(CoordinationStrategy):
    """Per-robot Kalman filter with innovation gating and belief reset.

    ``strategy_kwargs`` in the config are passed here, e.g.
    ``{gate_prob: 0.99, process_noise: 0.01, confirm: 1}`` (``gate`` sets the
    chi-square threshold directly and overrides ``gate_prob``)."""

    def __init__(self, gate: Optional[float] = None, gate_prob: float = 0.99,
                 process_noise: float = 0.01, confirm: int = 1):
        self.config = EstimatorConfig(process_noise=process_noise,
                                      gate=gate if gate is not None else chi2_threshold(gate_prob),
                                      confirm=confirm)

    def reset(self, robots, rng):
        super().reset(robots, rng)
        self.estimator = Estimator(self.config)

    def update_belief(self, robot, target_id, observation, neighbor_messages, t):
        prev = robot.beliefs.get(target_id, Belief())
        return self.estimator.update(robot.id, target_id, prev, observation, robot.sensor_noise_std, t)

    def detect_relocation(self, robot, target_id, observation, belief):
        return self.estimator.relocation_detected(robot.id, target_id)

    def get_messages_to_send(self, robot, comm, robots, t):
        return []


class IndependentStrategy(_SharedEstimatorBase):
    """Each robot tracks on its own; no communication.  Complete as it is."""
    name = "independent"


class CentralizedStrategy(_SharedEstimatorBase):
    """All observations are sent to a hub (e.g. robot 0) which fuses them and
    returns the fused estimate.
    TODO (research team): send ``messages.observation_message`` to the hub, fuse at
    the hub with ``self.estimator``, broadcast ``messages.belief_message`` back."""
    name = "centralized"
    hub_id = 0


class DecentralizedConsensusStrategy(_SharedEstimatorBase):
    """Neighbours exchange estimates and run a consensus update.

    Messaging is in place: every robot broadcasts its belief about every target to
    its graph neighbours each step (``messages.belief_message``).
    TODO (research team): the consensus update in ``update_belief`` -- use
    ``messages.for_target(neighbor_messages, target_id)``, ``messages.is_newer``
    and the estimator's gating to decide between fusing and resetting."""
    name = "decentralized"

    def get_messages_to_send(self, robot, comm, robots, t):
        return [(None, belief_message(robot.id, target_id, belief, t))
                for target_id, belief in robot.beliefs.items() if belief.valid and belief.cov is not None]


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
