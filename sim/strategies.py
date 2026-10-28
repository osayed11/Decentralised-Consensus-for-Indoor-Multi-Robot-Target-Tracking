"""Coordination strategies.

############################################################################
#  RESEARCH TEAM CONTRIBUTION -- STUBS ONLY.                               #
#                                                                          #
#  Each timestep the simulation, for every alive robot, calls:             #
#     1. msgs = robot.inbox (delivered since last step; cleared after)     #
#     2. for each target: belief = strategy.update_belief(robot, target_id,#
#            observation, msgs, t)                                         #
#     3. strategy.detect_relocation(robot, target_id, observation, belief) #
#     4. strategy.get_messages_to_send(robot, comm_layer, robots, t)       #
#        -> list of (receiver_id | None for broadcast, payload dict)       #
#                                                                          #
#  Placeholders implement "belief = most recent raw observation", never    #
#  flag a relocation, and send nothing.                                    #
############################################################################
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .comms import CommunicationLayer
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
# Trivial placeholder shared by the three stubs so the pipeline runs.
# ---------------------------------------------------------------------------
class _LastObservationStub(CoordinationStrategy):
    def update_belief(self, robot, target_id, observation, neighbor_messages, t):
        # TODO (research team): replace with real fusion logic.
        prev = robot.beliefs.get(target_id, Belief())
        if observation is not None:
            return Belief(mean=np.asarray(observation, dtype=float),
                          cov=np.eye(2) * robot.sensor_noise_std ** 2,
                          timestamp=t, source="own_obs")
        return prev

    def detect_relocation(self, robot, target_id, observation, belief):
        # TODO (research team): innovation gating.
        return False

    def get_messages_to_send(self, robot, comm, robots, t):
        # TODO (research team).
        return []


class IndependentStrategy(_LastObservationStub):
    """Each robot tracks on its own; no communication.  STUB."""
    name = "independent"


class CentralizedStrategy(_LastObservationStub):
    """All observations are sent to a hub (e.g. robot 0) which fuses them and
    returns the fused estimate.  STUB."""
    name = "centralized"
    hub_id = 0


class DecentralizedConsensusStrategy(_LastObservationStub):
    """Neighbours exchange estimates and run a consensus update.  STUB."""
    name = "decentralized"


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
