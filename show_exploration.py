#!/usr/bin/env python3
"""Run one independent trial and save a figure of each robot's private coverage.

Usage: python show_exploration.py --duration 60 --output results/coverage.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.harness import SimConfig, Simulation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='sim/configs/independent_exploration.yaml')
    ap.add_argument('--duration', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default='results/coverage.png')
    args = ap.parse_args()
    sim = Simulation(SimConfig.from_file(args.config), seed=args.seed)
    sim.run(time_limit=args.duration)
    n = len(sim.robots)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4), squeeze=False)
    for i, (robot, ax) in enumerate(zip(sim.robots, axes[0])):
        mem = sim.patrols[robot.id].memory
        # White = not yet observed; blue = observed; dark gray = obstacle/nonreachable.
        im = np.zeros(mem.reachable.shape, dtype=np.uint8)
        im[mem.reachable] = 1
        im[mem.observed_mask] = 2
        from matplotlib.colors import ListedColormap
        ax.imshow(im, origin='lower', interpolation='nearest',
                  cmap=ListedColormap(['#465363', '#f4f4f4', '#4e9bc0']),
                  extent=(0, sim.env.width, 0, sim.env.height), vmin=0, vmax=2)
        ax.scatter([robot.position[0]], [robot.position[1]], c='orange', s=50, edgecolors='black')
        ax.set_title(f'Robot {robot.id}: {100*mem.coverage_fraction:.1f}% observed')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    fig.suptitle(f'Private exploration histories at t={sim.t:.1f}s (seed={args.seed})')
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Saved {output}')
    print('Patrol goals assigned:', sum(p.state.goals_assigned for p in sim.patrols.values()))
    print('Team union observed fraction:', f'{sim.summary()["exploration_coverage_union"]:.3f}')


if __name__ == '__main__':
    main()
