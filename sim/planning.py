"""A* path planning on the occupancy grid.

``plan_path(start, goal, env) -> list of (x, y) waypoints`` in world metres.
Used for general robot movement (patrolling) and for dispatch navigation.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from .environment import Environment, Point

_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _nearest_free_cell(grid: np.ndarray, rc: Tuple[int, int], max_radius: int = 20) -> Optional[Tuple[int, int]]:
    r0, c0 = rc
    if not grid[r0, c0]:
        return rc
    rows, cols = grid.shape
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad):
                for r, c in ((r0 + dr, c0 + dc), (r0 + dc, c0 + dr)):
                    if 0 <= r < rows and 0 <= c < cols and not grid[r, c]:
                        return r, c
    return None


def astar_grid(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* over an 8-connected boolean grid (True = blocked). Returns cells."""
    rows, cols = grid.shape
    if grid[start] or grid[goal]:
        return None

    def h(a: Tuple[int, int]) -> float:
        dr, dc = abs(a[0] - goal[0]), abs(a[1] - goal[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)  # octile

    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    came = {}
    closed = set()
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        for dr, dc, cost in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
                continue  # no corner cutting
            ng = gc + cost
            nxt = (nr, nc)
            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def _simplify(env: Environment, pts: List[np.ndarray]) -> List[np.ndarray]:
    """Greedy string-pulling: skip intermediate waypoints with clear LOS on
    the inflated grid."""
    if len(pts) <= 2:
        return pts
    inflated_env = Environment(grid=env.inflated, resolution=env.resolution, robot_radius=0.0)
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not inflated_env.line_of_sight(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan_path(start: Point, goal: Point, env: Environment, simplify: bool = True) -> List[np.ndarray]:
    """Plan a collision-free path from ``start`` to ``goal`` (world metres).

    Returns a list of waypoints (excluding ``start``, ending at ``goal`` or the
    nearest free cell to it).  Returns an empty list if no path exists.
    """
    grid = env.inflated
    s = _nearest_free_cell(grid, env.world_to_grid(start))
    gl = _nearest_free_cell(grid, env.world_to_grid(goal))
    if s is None or gl is None:
        return []
    cells = astar_grid(grid, s, gl)
    if cells is None:
        return []
    pts = [np.asarray(start, dtype=float)] + [env.grid_to_world(c) for c in cells[1:]]
    if env.is_free(goal):
        pts[-1] = np.asarray(goal, dtype=float)
    if simplify:
        pts = _simplify(env, pts)
    return pts[1:]


def path_length(points: List[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))
