"""Floor plan, obstacles and occupancy grid.

The world is a rectangle ``[0, width] x [0, height]`` in metres.  The occupancy
grid is a boolean numpy array indexed ``grid[row, col]`` where ``True`` means
*occupied* (obstacle / wall).  Row 0 corresponds to ``y = 0`` (bottom).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


def default_floor_plan(resolution: float = 0.25) -> np.ndarray:
    """A simple 20 m x 14 m house/office: four rooms off a corridor.

    Returns an occupancy grid (True = obstacle) at the given resolution.
    """
    width_m, height_m = 20.0, 14.0
    rows = int(round(height_m / resolution))
    cols = int(round(width_m / resolution))
    grid = np.zeros((rows, cols), dtype=bool)

    def fill(x0: float, y0: float, x1: float, y1: float) -> None:
        c0, c1 = int(x0 / resolution), int(np.ceil(x1 / resolution))
        r0, r1 = int(y0 / resolution), int(np.ceil(y1 / resolution))
        grid[max(r0, 0):min(r1, rows), max(c0, 0):min(c1, cols)] = True

    wall = 0.3
    # Boundary walls
    fill(0, 0, width_m, wall)
    fill(0, height_m - wall, width_m, height_m)
    fill(0, 0, wall, height_m)
    fill(width_m - wall, 0, width_m, height_m)

    # Corridor spans y in [5.8, 8.2] (2.4 m clear). Bottom rooms y in [0, 5.5],
    # top rooms y in [8.5, 14].
    fill(0, 5.5, width_m, 5.5 + wall)
    fill(0, 8.5 - wall, width_m, 8.5)

    # Vertical walls splitting bottom into 3 rooms, top into 2 rooms
    fill(7.0, 0, 7.0 + wall, 5.5)
    fill(13.0, 0, 13.0 + wall, 5.5)
    fill(10.0, 8.5, 10.0 + wall, height_m)

    def carve(x0: float, y0: float, x1: float, y1: float) -> None:
        c0, c1 = int(x0 / resolution), int(np.ceil(x1 / resolution))
        r0, r1 = int(y0 / resolution), int(np.ceil(y1 / resolution))
        grid[max(r0, 0):min(r1, rows), max(c0, 0):min(c1, cols)] = False

    # Doors (1.8 m wide)
    carve(2.6, 5.5, 4.4, 5.5 + wall)      # room A door
    carve(9.1, 5.5, 10.9, 5.5 + wall)     # room B door
    carve(15.6, 5.5, 17.4, 5.5 + wall)    # room C door
    carve(4.1, 8.5 - wall, 5.9, 8.5)      # room D door
    carve(14.1, 8.5 - wall, 15.9, 8.5)    # room E door

    # Some furniture-like obstacles
    fill(1.5, 1.5, 3.5, 3.0)
    fill(15.0, 2.0, 18.0, 3.0)
    fill(2.0, 10.5, 4.0, 12.5)
    fill(12.0, 11.0, 13.5, 13.0)
    return grid


@dataclass
class Environment:
    """2D world with an occupancy grid.

    Parameters
    ----------
    grid : bool array (rows, cols), True = obstacle.  If None, the default
        floor plan is generated.
    resolution : metres per grid cell.
    robot_radius : used to inflate obstacles for planning / free-space checks.
    """

    grid: Optional[np.ndarray] = None
    resolution: float = 0.25
    robot_radius: float = 0.3
    inflated: np.ndarray = field(init=False, repr=False)
    reachable: np.ndarray = field(init=False, repr=False)  # largest connected free region

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = default_floor_plan(self.resolution)
        self.grid = np.asarray(self.grid, dtype=bool)
        self._ensure_boundary()
        self.inflated = self._inflate(self.grid, self.robot_radius)
        self.reachable, self.n_free_components = self._largest_component(~self.inflated)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_array(cls, arr: Iterable[Iterable[int]], resolution: float = 0.25, **kw) -> "Environment":
        """Build from a 2D array of 0/1 (1 = obstacle).  Row 0 is the bottom."""
        return cls(grid=np.asarray(arr, dtype=bool), resolution=resolution, **kw)

    @classmethod
    def from_image(cls, path: str, resolution: float = 0.25, threshold: int = 128, **kw) -> "Environment":
        """Build from an image mask: dark pixels are obstacles. Image row 0 is
        the top, so it is flipped so that row 0 of the grid is the bottom."""
        import matplotlib.image as mpimg

        img = mpimg.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
        if img.max() <= 1.0:
            img = img * 255
        grid = img < threshold
        return cls(grid=np.flipud(grid), resolution=resolution, **kw)

    @classmethod
    def from_size(cls, width_m: float, height_m: float, resolution: float = 0.25, **kw) -> "Environment":
        """Empty world of the given size with boundary walls only."""
        rows, cols = int(round(height_m / resolution)), int(round(width_m / resolution))
        return cls(grid=np.zeros((rows, cols), dtype=bool), resolution=resolution, **kw)

    # ---- geometry -----------------------------------------------------
    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def width(self) -> float:
        return self.cols * self.resolution

    @property
    def height(self) -> float:
        return self.rows * self.resolution

    def world_to_grid(self, p: Point) -> Tuple[int, int]:
        c = int(np.clip(p[0] / self.resolution, 0, self.cols - 1))
        r = int(np.clip(p[1] / self.resolution, 0, self.rows - 1))
        return r, c

    def grid_to_world(self, rc: Tuple[int, int]) -> np.ndarray:
        r, c = rc
        return np.array([(c + 0.5) * self.resolution, (r + 0.5) * self.resolution])

    def in_bounds(self, p: Point) -> bool:
        return 0.0 <= p[0] < self.width and 0.0 <= p[1] < self.height

    def is_free(self, p: Point, inflated: bool = True) -> bool:
        if not self.in_bounds(p):
            return False
        r, c = self.world_to_grid(p)
        g = self.inflated if inflated else self.grid
        return not g[r, c]

    def is_reachable(self, p: Point) -> bool:
        """Free and inside the largest connected free region."""
        if not self.in_bounds(p):
            return False
        r, c = self.world_to_grid(p)
        return bool(self.reachable[r, c])

    def random_free_position(self, rng: np.random.Generator, min_dist_from: Iterable[Point] = (),
                             min_dist: float = 0.0, max_tries: int = 10_000) -> np.ndarray:
        """Uniform random position in the largest connected free region, so
        every sampled robot / target / goal position is mutually reachable."""
        others = [np.asarray(o) for o in min_dist_from]
        for _ in range(max_tries):
            p = rng.uniform([0, 0], [self.width, self.height])
            if not self.is_reachable(p):
                continue
            if others and any(np.linalg.norm(p - o) < min_dist for o in others):
                continue
            return p
        raise RuntimeError("Could not find a free position")

    def line_of_sight(self, a: Point, b: Point, inflated: bool = False) -> bool:
        """Supercover raycast (Amanatides-Woo grid traversal): visits *every*
        cell the segment a->b passes through, so a continuous mover that
        follows the segment never enters an occupied cell.  Uses the raw grid
        by default (sensor model add-on) or the inflated grid when
        ``inflated=True`` (robot-body clearance checks)."""
        grid = self.inflated if inflated else self.grid
        res = self.resolution
        x0, y0 = float(a[0]) / res, float(a[1]) / res
        x1, y1 = float(b[0]) / res, float(b[1]) / res
        c, r = int(np.floor(x0)), int(np.floor(y0))
        c1, r1 = int(np.floor(x1)), int(np.floor(y1))
        rows, cols = grid.shape
        clip = lambda rr, cc: (min(max(rr, 0), rows - 1), min(max(cc, 0), cols - 1))
        rr, cc = clip(r, c)
        if grid[rr, cc]:
            return False
        dx, dy = x1 - x0, y1 - y0
        step_c = 1 if dx > 0 else -1
        step_r = 1 if dy > 0 else -1
        t_max_x = ((c + (step_c > 0)) - x0) / dx if dx != 0 else np.inf
        t_max_y = ((r + (step_r > 0)) - y0) / dy if dy != 0 else np.inf
        t_dx = abs(1.0 / dx) if dx != 0 else np.inf
        t_dy = abs(1.0 / dy) if dy != 0 else np.inf
        for _ in range(2 * (abs(c1 - c) + abs(r1 - r)) + 2):
            if c == c1 and r == r1:
                break
            if t_max_x < t_max_y:
                c += step_c
                t_max_x += t_dx
            elif t_max_y < t_max_x:
                r += step_r
                t_max_y += t_dy
            else:  # exact corner crossing: check both side cells (conservative)
                if grid[clip(r, c + step_c)] or grid[clip(r + step_r, c)]:
                    return False
                c += step_c
                r += step_r
                t_max_x += t_dx
                t_max_y += t_dy
            if grid[clip(r, c)]:
                return False
        return True

    # ---- internals ----------------------------------------------------
    def _ensure_boundary(self) -> None:
        self.grid[0, :] = True
        self.grid[-1, :] = True
        self.grid[:, 0] = True
        self.grid[:, -1] = True

    @staticmethod
    def _largest_component(free: np.ndarray):
        """4-connected flood fill; returns (mask of largest free component, n components)."""
        from collections import deque

        rows, cols = free.shape
        labels = np.zeros_like(free, dtype=np.int32)
        best, best_size, n = 0, 0, 0
        for r0, c0 in zip(*np.nonzero(free)):
            if labels[r0, c0]:
                continue
            n += 1
            labels[r0, c0] = n
            q, size = deque([(r0, c0)]), 1
            while q:
                r, c = q.popleft()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = n
                        size += 1
                        q.append((nr, nc))
            if size > best_size:
                best, best_size = n, size
        return labels == best, n

    def _inflate(self, grid: np.ndarray, radius: float) -> np.ndarray:
        """Mark every cell whose centre lies within ``radius`` of an occupied
        cell centre (exact distance, so 0.3 m at 0.25 m/cell = 1 cell)."""
        rk = radius / self.resolution
        k = int(np.floor(rk))
        if k <= 0:
            return grid.copy()
        out = grid.copy()
        rows, cols = grid.shape
        occ_r, occ_c = np.nonzero(grid)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr * dr + dc * dc > rk * rk:
                    continue
                rr = np.clip(occ_r + dr, 0, rows - 1)
                cc = np.clip(occ_c + dc, 0, cols - 1)
                out[rr, cc] = True
        return out
