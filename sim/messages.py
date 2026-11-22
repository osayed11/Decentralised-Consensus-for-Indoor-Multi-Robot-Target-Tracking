"""Agreed message formats (see SHARED_COMPONENTS.md).

Payloads are plain dicts so ``CommunicationLayer`` can size and log them.  Every
payload has ``type``, ``sender``, ``target_id`` and ``t_sent``.

``belief``       a robot's current estimate.  Decentralized: sent to neighbours
                 every step.  Centralized: the hub sends the fused estimate back.
    mean (2,), cov (2, 2), last_seen, reset_count, last_absence
``observation``  a raw measurement.  Centralized: robots send these to the hub.
    z (2,), noise_std, t_obs
``absence``      "I looked where the target should be and it was not there"
                 (no-sighting rule).  The robot has cleared its belief, so this
                 is what it shares instead.  Decentralized: sent to neighbours
                 while the robot has no belief.  Centralized: robots tell the
                 server, which cannot see absences otherwise.
    pos (2,) = centre of the area seen empty, radius, t_obs = when

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
ABSENCE = "absence"


def belief_message(sender_id: int, target_id: int, belief: Belief, t: float) -> dict:
    return {"type": BELIEF, "sender": sender_id, "target_id": target_id, "t_sent": t,
            "mean": np.array(belief.mean, dtype=float), "cov": np.array(belief.cov, dtype=float),
            "last_seen": float(belief.last_seen), "reset_count": int(belief.reset_count),
            "last_absence": float(belief.last_absence)}


def observation_message(sender_id: int, target_id: int, z: np.ndarray, noise_std: float, t: float) -> dict:
    return {"type": OBSERVATION, "sender": sender_id, "target_id": target_id, "t_sent": t,
            "z": np.array(z, dtype=float), "noise_std": float(noise_std), "t_obs": t}


def absence_message(sender_id: int, target_id: int, belief: Belief, t: float) -> dict:
    """Share a cleared belief's empty check: the area (``absence_pos``,
    ``absence_radius``) was seen empty at ``last_absence``."""
    return {"type": ABSENCE, "sender": sender_id, "target_id": target_id, "t_sent": t,
            "pos": np.array(belief.absence_pos, dtype=float), "radius": float(belief.absence_radius),
            "t_obs": float(belief.last_absence)}


def for_target(messages: Iterable[dict], target_id: int, kind: str = BELIEF) -> List[dict]:
    """The messages of one type about one target (an inbox holds all targets and types)."""
    return [m for m in messages if m.get("type") == kind and m.get("target_id") == target_id]


def message_to_belief(msg: dict) -> Belief:
    """A received belief message as a Belief (source='msg')."""
    return Belief(mean=np.array(msg["mean"], dtype=float), cov=np.array(msg["cov"], dtype=float),
                  timestamp=msg["t_sent"], source="msg", last_seen=msg["last_seen"],
                  reset_count=msg["reset_count"], last_absence=msg.get("last_absence", -np.inf))


def is_newer(msg: dict, belief: Belief) -> bool:
    """True if the message is based on a more recent sighting than ``belief``."""
    return (not belief.valid) or msg["last_seen"] > belief.last_seen


def contradicted_by_absence(claim_pos: np.ndarray, claim_last_seen: float, checked_pos: np.ndarray,
                            checked_at: float, radius: float) -> bool:
    """True if a belief that the target is at ``claim_pos`` (based on a sighting
    at ``claim_last_seen``) is contradicted by an area (centre ``checked_pos``,
    ``radius``) seen empty at ``checked_at``: the claim is older than the check
    and points inside that area.  Use it both ways in a consensus
    step: to ignore a neighbour's stale belief, and to adopt a neighbour's
    "not there" when my own belief points at the spot they found empty.
    A sighting *elsewhere* is never contradicted, nor one newer than the check."""
    if not np.isfinite(checked_at) or claim_last_seen >= checked_at:
        return False
    return float(np.linalg.norm(np.asarray(claim_pos) - np.asarray(checked_pos))) <= radius


def belief_contradicted(claim_pos: np.ndarray, claim_last_seen: float, belief: Belief) -> bool:
    """``contradicted_by_absence`` against the empty check stored in ``belief``."""
    if belief.absence_pos is None:
        return False
    return contradicted_by_absence(claim_pos, claim_last_seen, belief.absence_pos,
                                   belief.last_absence, belief.absence_radius)
