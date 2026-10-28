"""Fault injection infrastructure.

* Scheduled robot failure: robot stops sensing, moving and communicating from
  timestep ``t`` onward.
* Packet loss probability is a *parameter* only: it is plumbed into the comms
  stub (research team) which is responsible for applying it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .robot import Robot


@dataclass
class FaultSchedule:
    robot_failures: Dict[int, float] = field(default_factory=dict)  # robot_id -> failure time (s)
    packet_loss_prob: float = 0.0
    log: List[dict] = field(default_factory=list)

    def apply(self, t: float, robots: List[Robot]) -> None:
        for r in robots:
            ft = self.robot_failures.get(r.id)
            if ft is not None and not r.failed and t >= ft:
                r.fail(t)
                self.log.append({"t": t, "event": "robot_failure", "robot_id": r.id})
