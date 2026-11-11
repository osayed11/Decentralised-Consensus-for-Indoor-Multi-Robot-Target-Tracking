"""Communication graph and message passing (shared by all strategies).

Every step the simulation calls ``build_graph``; strategies then send with
``send_message`` / ``broadcast``.  Delivered payloads land in the receiver's
``robot.inbox`` and are read by the strategy on the *next* step.

* Link model: an edge (i, j) exists iff both robots are alive and their
  distance is <= min(comm_range_i, comm_range_j) (symmetric links).
* Each transmission over an existing link is lost with probability
  ``packet_loss_prob``.  The loss draws come from the layer's own random
  stream, seeded once from the simulation RNG, so how many messages a
  strategy sends never changes the rest of the world (target relocations,
  patrol goals): every strategy is compared on the same scenario.
* A broadcast is counted as one transmission per neighbour (unicast model).
* Every attempt is logged as a ``MessageRecord`` (delivered or not, and why)
  for the communication-cost and robustness metrics.

Payload formats are defined in ``sim/messages.py``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from .robot import Robot


@dataclass
class MessageRecord:
    t: float
    sender: int
    receiver: int
    payload_size: int
    delivered: bool
    reason: str = ""  # "delivered" | "lost" (packet loss) | "no_link" (out of range / failed)


@dataclass
class CommunicationLayer:
    packet_loss_prob: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    graph: nx.Graph = field(default_factory=nx.Graph)
    log: List[MessageRecord] = field(default_factory=list)
    _t: float = 0.0

    def __post_init__(self) -> None:
        # Independent stream for packet-loss draws (see module docstring).
        self.rng = np.random.default_rng(int(self.rng.integers(2**63 - 1)))

    # -- called by the simulation every timestep --------------------------
    def build_graph(self, robots: List[Robot], t: float) -> nx.Graph:
        """Edge between every pair of alive robots within communication range."""
        self._t = t
        g = nx.Graph()
        alive = [r for r in robots if not r.failed]
        g.add_nodes_from(r.id for r in alive)
        for a_index, a in enumerate(alive):
            for b in alive[a_index + 1:]:
                if np.linalg.norm(a.position - b.position) <= min(a.comm_range, b.comm_range):
                    g.add_edge(a.id, b.id)
        self.graph = g
        return g

    def neighbors(self, robot_id: int) -> List[int]:
        return list(self.graph.neighbors(robot_id)) if robot_id in self.graph else []

    # -- called by strategies ---------------------------------------------
    def send_message(self, sender: Robot, receiver: Robot, payload: Dict[str, Any]) -> bool:
        """Deliver a copy of ``payload`` into ``receiver.inbox`` if the link exists
        and the packet is not lost.  Every attempt is logged.  Returns delivered."""
        size = _payload_size(payload)
        linked = (not sender.failed and not receiver.failed
                  and self.graph.has_edge(sender.id, receiver.id))
        if not linked:
            self.log.append(MessageRecord(self._t, sender.id, receiver.id, size, False, "no_link"))
            return False
        if self.packet_loss_prob > 0 and self.rng.random() < self.packet_loss_prob:
            self.log.append(MessageRecord(self._t, sender.id, receiver.id, size, False, "lost"))
            return False
        receiver.inbox.append(copy.deepcopy(payload))  # receivers must not share arrays with the sender
        self.log.append(MessageRecord(self._t, sender.id, receiver.id, size, True, "delivered"))
        return True

    def broadcast(self, sender: Robot, robots: List[Robot], payload: Dict[str, Any]) -> int:
        """Send to every current graph neighbour of ``sender``.  Returns deliveries."""
        by_id = {r.id: r for r in robots}
        return sum(int(self.send_message(sender, by_id[nid], payload)) for nid in self.neighbors(sender.id))

    # -- metrics -----------------------------------------------------------
    def messages_sent(self) -> int:
        return len(self.log)

    def messages_delivered(self) -> int:
        return sum(1 for m in self.log if m.delivered)

    def messages_lost(self) -> int:
        """Transmissions over an existing link that were dropped by packet loss."""
        return sum(1 for m in self.log if m.reason == "lost")

    def bytes_sent(self) -> int:
        return sum(m.payload_size for m in self.log)

    def reset_log(self) -> None:
        self.log.clear()


def _payload_size(payload: Dict[str, Any]) -> int:
    """Rough byte count of a payload for the communication-cost metric."""
    size = 0
    for v in payload.values():
        if isinstance(v, np.ndarray):
            size += v.nbytes
        elif isinstance(v, (int, float, bool)):
            size += 8
        elif isinstance(v, str):
            size += len(v.encode())
        elif isinstance(v, dict):
            size += _payload_size(v)
        else:
            size += 8
    return size
