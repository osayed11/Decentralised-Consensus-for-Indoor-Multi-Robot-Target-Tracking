"""Communication-free local geometric robot collision prevention.

Treat other robots as stationary disks during the current short motion step.
The swept segment is clipped *before* reaching the disk's safety boundary;
robots must be within the local proximity sensing radius to be considered.
This deliberately does not use the communication graph, inbox, or beliefs.
"""
from __future__ import annotations

import numpy as np


def safe_motion_fraction(start: np.ndarray, displacement: np.ndarray,
                         obstacles: list[np.ndarray], clearance: float) -> float:
    """Largest safe prefix of a straight motion, avoiding other robot centres.

    Returns a value in [0, 1].  Uses an exact ray-circle entry intersection,
    rather than checking only the endpoint. `clearance` is centre-to-centre.
    A robot already at/inside clearance may move *away* but not closer.
    """
    start = np.asarray(start, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    length_sq = float(displacement @ displacement)
    if length_sq <= 1e-20:
        return 1.0
    fraction = 1.0
    for other in obstacles:
        relative = start - np.asarray(other, dtype=float)
        b = float(relative @ displacement)
        c = float(relative @ relative - clearance * clearance)
        if c <= 1e-10:
            # Already on/inside the safety surface: allow only non-approaching
            # motion, and only when the swept segment never enters the disk.
            if b < -1e-12:
                return 0.0
            continue
        if b >= 0:  # moving initially away; no forward intersection
            continue
        disc = b * b - length_sq * c
        if disc < 0:
            continue
        entry = (-b - np.sqrt(max(0.0, disc))) / length_sq
        if 0 <= entry <= fraction:
            # Small buffer to remain outside the separation boundary.
            fraction = max(0.0, float(entry) - 1e-7)
    return float(np.clip(fraction, 0.0, 1.0))
