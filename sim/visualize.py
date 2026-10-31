"""Matplotlib 2D view, animation export and result plots."""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse

from .harness import Simulation

ROBOT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _cov_ellipse(mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, **kw) -> Ellipse:
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)


def draw_frame(sim: Simulation, ax: Optional[plt.Axes] = None, show_comm: bool = True,
               show_sensor: bool = True, show_belief: bool = True, show_paths: bool = False) -> plt.Axes:
    """Render the current simulation state onto ``ax``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))
    ax.clear()
    env = sim.env
    ax.imshow(env.grid, cmap="Greys", origin="lower", extent=(0, env.width, 0, env.height),
              interpolation="nearest", alpha=0.9, zorder=0)

    # comm graph edges
    if show_comm:
        pos = {r.id: r.position for r in sim.robots}
        for i, j in sim.comm.graph.edges:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="tab:green", lw=1.2, alpha=0.7, zorder=2)

    for tg in sim.targets:
        ax.plot(*tg.position, marker="*", ms=16, color="gold", mec="k", zorder=6, label=f"target {tg.id}" if tg.id == 0 else None)

    for r in sim.robots:
        c = ROBOT_COLORS[r.id % len(ROBOT_COLORS)]
        if show_sensor and not r.failed:
            ax.add_patch(Circle(r.position, r.sensor_range, fc=c, ec=c, alpha=0.08, lw=0.8, zorder=1))
        if show_paths and r.path:
            pts = np.vstack([r.position] + r.path)
            ax.plot(pts[:, 0], pts[:, 1], "--", color=c, lw=0.8, alpha=0.6, zorder=3)
        marker = "x" if r.failed else ("s" if r.mode == "dispatch" else "o")
        ax.plot(*r.position, marker=marker, ms=9, color=c, mec="k", zorder=5)
        ax.annotate(str(r.id), r.position, xytext=(5, 5), textcoords="offset points", fontsize=8, color=c)
        hx = r.position + 0.5 * np.array([np.cos(r.heading), np.sin(r.heading)])
        ax.plot([r.position[0], hx[0]], [r.position[1], hx[1]], color=c, lw=1.5, zorder=5)
        if show_belief:
            for tid, b in r.beliefs.items():
                if not b.valid:
                    continue
                ax.plot(*b.mean, marker="+", ms=10, color=c, mew=1.8, zorder=4)
                if b.cov is not None:
                    ax.add_patch(_cov_ellipse(b.mean, b.cov, fc="none", ec=c, lw=1.0, ls=":", zorder=4))

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    active = ", ".join(f"T{tid}->R{ad.robot.id}" for tid, ad in sim.dispatcher.active.items())
    ax.set_title(f"t = {sim.t:6.1f} s   strategy = {sim.strategy.name}   msgs = {sim.comm.messages_sent()}"
                 + (f"   dispatch: {active}" if active else ""), fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


def animate(sim: Simulation, duration: Optional[float] = None, steps_per_frame: int = 1, fps: int = 20,
            save_path: Optional[str] = None, show: bool = True, **draw_kw):
    """Run the simulation while rendering. If ``save_path`` ends in .mp4 or
    .gif the animation is written to disk (ffmpeg / pillow required).

    Playback speed = steps_per_frame * dt * fps  (defaults: 1 * 0.1 * 20 = 2x real time)."""
    from matplotlib.animation import FuncAnimation

    duration = sim.cfg.time_limit if duration is None else duration
    n_frames = int(duration / (sim.cfg.dt * steps_per_frame))
    fig, ax = plt.subplots(figsize=(9, 6.5))

    def update(_frame):
        for _ in range(steps_per_frame):
            if sim.t < duration:
                sim.step()
        draw_frame(sim, ax, **draw_kw)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False, repeat=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps)
        print(f"saved animation to {save_path}")
    if show:
        plt.show()
    return anim


def export_frames(sim: Simulation, out_dir: str, every_s: float = 1.0, duration: Optional[float] = None, **draw_kw) -> List[str]:
    """Save PNG frames every ``every_s`` seconds of sim time (for video demos)."""
    os.makedirs(out_dir, exist_ok=True)
    duration = sim.cfg.time_limit if duration is None else duration
    stride = max(1, int(round(every_s / sim.cfg.dt)))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    paths = []
    while sim.t < duration:
        if sim.step_idx % stride == 0:
            draw_frame(sim, ax, **draw_kw)
            p = os.path.join(out_dir, f"frame_{sim.step_idx:06d}.png")
            fig.savefig(p, dpi=100)
            paths.append(p)
        sim.step()
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Result plots (consume harness logs)
# ---------------------------------------------------------------------------
def plot_error_over_time(step_df: pd.DataFrame, ax: Optional[plt.Axes] = None, per_robot: bool = True,
                         events_df: Optional[pd.DataFrame] = None) -> plt.Axes:
    """Belief error vs time for one trial (team-min and per-robot)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    team = step_df.groupby("t")["error"].min()
    ax.plot(team.index, team.values, color="k", lw=1.8, label="team (best robot)")
    if per_robot:
        for rid, g in step_df.groupby("robot_id"):
            ax.plot(g["t"], g["error"], lw=0.8, alpha=0.6, label=f"robot {rid}")
    if events_df is not None and not events_df.empty:
        for t in events_df.loc[events_df["event"] == "relocation", "t"]:
            ax.axvline(t, color="tab:red", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("belief error [m]")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    return ax


def plot_rmse_over_time(trials_dir: str, ax: Optional[plt.Axes] = None, label: Optional[str] = None) -> plt.Axes:
    """RMSE of team error vs time averaged across all trial step logs in a results dir."""
    import glob

    files = sorted(glob.glob(os.path.join(trials_dir, "trial_*_steps.csv")))
    series = []
    for f in files:
        df = pd.read_csv(f)
        series.append(df.groupby("t")["error"].min() ** 2)
    if not series:
        raise FileNotFoundError(f"no step logs in {trials_dir}")
    mse = pd.concat(series, axis=1).mean(axis=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(mse.index, np.sqrt(mse.values), label=label or os.path.basename(trials_dir))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("team RMSE [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_metric_bars(trials: pd.DataFrame, metric: str, group: str = "strategy", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Bar chart of mean +- 95% CI of ``metric`` per ``group`` (e.g. search
    efficiency: time_to_first_localisation per strategy)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    g = trials.groupby(group)[metric]
    mean, n, std = g.mean(), g.count(), g.std(ddof=1)
    ci = 1.96 * std / np.sqrt(n)
    ax.bar(mean.index.astype(str), mean.values, yerr=ci.values, capsize=4, alpha=0.85)
    ax.set_ylabel(metric)
    ax.set_xlabel(group)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_degradation(sweep: pd.DataFrame, metric: str, ax: Optional[plt.Axes] = None, hue: Optional[str] = "strategy") -> plt.Axes:
    """Robustness curve: ``metric`` vs the sweep value (e.g. packet_loss_prob
    or n_robots), one line per ``hue``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    groups = sweep.groupby(hue) if hue and hue in sweep else [(None, sweep)]
    for name, g in groups:
        agg = g.groupby("sweep_value")[metric].agg(["mean", "std", "count"])
        ci = 1.96 * agg["std"] / np.sqrt(agg["count"])
        ax.errorbar(agg.index, agg["mean"], yerr=ci, marker="o", capsize=3, label=str(name) if name is not None else None)
    ax.set_xlabel(sweep["sweep_param"].iloc[0] if "sweep_param" in sweep else "value")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    if hue and hue in sweep:
        ax.legend(fontsize=8)
    return ax
