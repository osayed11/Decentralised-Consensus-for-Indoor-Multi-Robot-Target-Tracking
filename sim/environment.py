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

    # Horizontal corridor walls: corridor spans y in [6, 8]
    # Bottom rooms occupy y in [0, 6], top rooms y in [8, 14]
    # Wall between bottom rooms and corridor, with doors
    fill(0, 6.0, width_m, 6.0 + wall)
    fill(0, 8.0 - wall, width_m, 8.0)

    # Vertical walls splitting bottom into 3 rooms, top into 2 rooms
    fill(7.0, 0, 7.0 + wall, 6.0)
    fill(13.0, 0, 13.0 + wall, 6.0)
    fill(10.0, 8.0, 10.0 + wall, height_m)

    # Doors (carve free space): bottom rooms
    def carve(x0: float, y0: float, x1: float, y1: float) -> None:
        c0, c1 = int(x0 / resolution), int(np.ceil(x1 / resolution))
        r0, r1 = int(y0 / resolution), int(np.ceil(y1 / resolution))
        grid[max(r0, 0):min(r1, rows), max(c0, 0):min(c1, cols)] = False

    carve(3.0, 6.0, 4.5, 6.0 + wall)      # room A door
    carve(9.5, 6.0, 11.0, 6.0 + wall)     # room B door
    carve(16.0, 6.0, 17.5, 6.0 + wall)    # room C door
    carve(4.0, 8.0 - wall, 5.5, 8.0)      # room D door
    carve(14.5, 8.0 - wall, 16.0, 8.0)    # room E door

    # Some furniture-like obstacles
    fill(1.5, 1.5, 3.5, 3.0)
    fill(15.0, 2.0, 18.0, 3.0)
    fill(2.0, 10.0, 4.0, 12.5)
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

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = default_floor_plan(self.resolution)
        self.grid = np.asarray(self.grid, dtype=bool)
        self._ensure_boundary()
        self.inflated = self._inflate(self.grid, self.robot_radius)

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

    def random_free_position(self, rng: np.random.Generator, min_dist_from: Iterable[Point] = (),
                             min_dist: float = 0.0, max_tries: int = 10_000) -> np.ndarray:
        """Uniform random position in (inflated) free space."""
        others = [np.asarray(o) for o in min_dist_from]
        for _ in range(max_tries):
            p = rng.uniform([0, 0], [self.width, self.height])
            if not self.is_free(p):
                continue
            if others and any(np.linalg.norm(p - o) < min_dist for o in others):
                continue
            return p
        raise RuntimeError("Could not find a free position")

    def line_of_sight(self, a: Point, b: Point) -> bool:
        """Bresenham raycast on the (non-inflated) grid. True if no obstacle
        between a and b.  Optional add-on for the sensor model."""
        r0, c0 = self.world_to_grid(a)
        r1, c1 = self.world_to_grid(b)
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err = dc - dr
        r, c = r0, c0
        while True:
            if self.grid[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    # ---- internals ----------------------------------------------------
    def _ensure_boundary(self) -> None:
        self.grid[0, :] = True
        self.grid[-1, :] = True
        self.grid[:, 0] = True
        self.grid[:, -1] = True

    def _inflate(self, grid: np.ndarray, radius: float) -> np.ndarray:
        k = int(np.ceil(radius / self.resolution))
        if k <= 0:
            return grid.copy()
        out = grid.copy()
        rows, cols = grid.shape
        occ_r, occ_c = np.nonzero(grid)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr * dr + dc * dc > k * k:
                    continue
                rr = np.clip(occ_r + dr, 0, rows - 1)
                cc = np.clip(occ_c + dc, 0, cols - 1)
                out[rr, cc] = True
        return out
