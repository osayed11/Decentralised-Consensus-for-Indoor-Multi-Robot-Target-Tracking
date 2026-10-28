"""Communication graph and message passing.

############################################################################
#  RESEARCH TEAM CONTRIBUTION -- STUB ONLY.                                #
#                                                                          #
#  The simulation calls ``CommunicationLayer.build_graph`` once per step   #
#  and ``send_message`` / ``broadcast`` when a strategy wants to talk.     #
#  The placeholder below provides the *interface* and a trivial behaviour  #
#  (fully-connected-within-range graph is NOT assumed; the stub returns    #
#  a graph with no edges and drops every message) so the pipeline runs.   #
#                                                                          #
#  To implement:                                                           #
#   * build_graph: edge (i, j) iff distance(i, j) <= comm_range (and both   #
#     robots alive).  Return a networkx.Graph (or adjacency dict).         #
#   * send_message / broadcast: respect the graph, apply packet loss with  #
#     probability ``packet_loss_prob``, deliver into ``robot.inbox`` and   #
#     append a record to ``self.log`` for the communication-cost metric.   #
#   * Delivered messages are read by strategies on the *next* step via     #
#     ``neighbor_messages``.                                               #
############################################################################
"""
from __future__ import annotations

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


@dataclass
class CommunicationLayer:
    packet_loss_prob: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    graph: nx.Graph = field(default_factory=nx.Graph)
    log: List[MessageRecord] = field(default_factory=list)
    _t: float = 0.0

    # -- called by the simulation every timestep --------------------------
    def build_graph(self, robots: List[Robot], t: float) -> nx.Graph:
        """TODO (research team): add edges for robot pairs within comm_range.

        Placeholder: nodes only, no edges."""
        self._t = t
        g = nx.Graph()
        g.add_nodes_from(r.id for r in robots if not r.failed)
        self.graph = g
        return g

    def neighbors(self, robot_id: int) -> List[int]:
        return list(self.graph.neighbors(robot_id)) if robot_id in self.graph else []

    # -- called by strategies ---------------------------------------------
    def send_message(self, sender: Robot, receiver: Robot, payload: Dict[str, Any]) -> bool:
        """TODO (research team): deliver iff edge exists and packet not lost.

        Placeholder: log the attempt as undelivered and drop it."""
        self.log.append(MessageRecord(self._t, sender.id, receiver.id, _payload_size(payload), False))
        return False

    def broadcast(self, sender: Robot, robots: List[Robot], payload: Dict[str, Any]) -> int:
        """TODO (research team): send to every graph neighbour of ``sender``.

        Placeholder: no neighbours, so nothing is sent. Returns deliveries."""
        delivered = 0
        for nid in self.neighbors(sender.id):
            rcv = next(r for r in robots if r.id == nid)
            delivered += int(self.send_message(sender, rcv, payload))
        return delivered

    # -- metrics -----------------------------------------------------------
    def messages_sent(self) -> int:
        return len(self.log)

    def messages_delivered(self) -> int:
        return sum(1 for m in self.log if m.delivered)

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
