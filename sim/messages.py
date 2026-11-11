"""Agreed message formats (see SHARED_COMPONENTS.md).

Payloads are plain dicts so ``CommunicationLayer`` can size and log them.  Every
payload has ``type``, ``sender``, ``target_id`` and ``t_sent``.

``belief``       a robot's current estimate.  Decentralized: sent to neighbours
                 every step.  Centralized: the hub sends the fused estimate back.
    mean (2,), cov (2, 2), last_seen, reset_count
``observation``  a raw measurement.  Centralized: robots send these to the hub.
    z (2,), noise_std, t_obs

``last_seen`` is the time of the newest real observation behind a belief.  It is
what decides whose information is newer: a belief built on an older sighting must
not overrule one built after the target moved.  ``reset_count`` is a local counter
(each robot counts its own resets), so it is for diagnostics, not for comparison
between robots.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .robot import Belief

BELIEF = "belief"
OBSERVATION = "observation"


def belief_message(sender_id: int, target_id: int, belief: Belief, t: float) -> dict:
    return {"type": BELIEF, "sender": sender_id, "target_id": target_id, "t_sent": t,
            "mean": np.array(belief.mean, dtype=float), "cov": np.array(belief.cov, dtype=float),
            "last_seen": float(belief.last_seen), "reset_count": int(belief.reset_count)}


def observation_message(sender_id: int, target_id: int, z: np.ndarray, noise_std: float, t: float) -> dict:
    return {"type": OBSERVATION, "sender": sender_id, "target_id": target_id, "t_sent": t,
            "z": np.array(z, dtype=float), "noise_std": float(noise_std), "t_obs": t}


def for_target(messages: Iterable[dict], target_id: int, kind: str = BELIEF) -> List[dict]:
    """The messages of one type about one target (an inbox holds all targets and types)."""
    return [m for m in messages if m.get("type") == kind and m.get("target_id") == target_id]


def message_to_belief(msg: dict) -> Belief:
    """A received belief message as a Belief (source='msg')."""
    return Belief(mean=np.array(msg["mean"], dtype=float), cov=np.array(msg["cov"], dtype=float),
                  timestamp=msg["t_sent"], source="msg", last_seen=msg["last_seen"],
                  reset_count=msg["reset_count"])


def is_newer(msg: dict, belief: Belief) -> bool:
    """True if the message is based on a more recent sighting than ``belief``."""
    return (not belief.valid) or msg["last_seen"] > belief.last_seen
