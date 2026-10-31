#!/usr/bin/env python3
"""Visual sanity check: animate one trial.

  python demo.py                                 # live window
  python demo.py --save results/demo.gif         # export (pillow)
  python demo.py --save results/demo.mp4         # export (ffmpeg)
  python demo.py --frames results/frames         # PNG frames every 1 s
"""
import argparse

from sim.harness import SimConfig, Simulation
from sim.visualize import animate, export_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sim/configs/default.yaml")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--save", default=None, help=".gif or .mp4 path")
    ap.add_argument("--frames", default=None, help="directory for PNG frames")
    ap.add_argument("--paths", action="store_true", help="draw planned paths")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed relative to real time")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    cfg = SimConfig.from_file(args.config)
    if args.strategy:
        cfg = cfg.with_overrides(strategy=args.strategy)
    sim = Simulation(cfg, seed=args.seed)
    if args.frames:
        paths = export_frames(sim, args.frames, every_s=1.0, duration=args.duration, show_paths=args.paths)
        print(f"wrote {len(paths)} frames to {args.frames}")
    else:
        spf = max(1, int(round(args.speed / (cfg.dt * args.fps))))
        animate(sim, duration=args.duration, steps_per_frame=spf, fps=args.fps, save_path=args.save,
                show=args.save is None, show_paths=args.paths)


if __name__ == "__main__":
    main()
