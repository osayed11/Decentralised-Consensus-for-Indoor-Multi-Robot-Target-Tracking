#!/usr/bin/env python3
"""Run trials from a config file.

Examples
--------
  python run_experiments.py --config sim/configs/default.yaml
  python run_experiments.py --config sim/configs/default.yaml --trials 10 --strategy centralized
  python run_experiments.py --config sim/configs/scalability.yaml --sweep n_robots 2,4,6,8
  python run_experiments.py --config sim/configs/faults.yaml --sweep packet_loss_prob 0,0.2,0.5,0.8
  python run_experiments.py --config sim/configs/default.yaml --sweep strategy independent,centralized,decentralized
"""
import argparse

from sim.harness import SimConfig, aggregate, run_sweep, run_trials


def _parse_value(param: str, s: str):
    for cast in (int, float):
        try:
            v = cast(s)
            return v if param != "n_robots" else int(v)
        except ValueError:
            pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sim/configs/default.yaml")
    ap.add_argument("--trials", type=int, default=None, help="override n_trials")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--n-robots", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sweep", nargs=2, metavar=("PARAM", "VALUES"), help="e.g. --sweep n_robots 2,4,8")
    ap.add_argument("--no-step-logs", action="store_true", help="skip per-step CSVs (faster, smaller)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig.from_file(args.config)
    overrides = {k: v for k, v in {"strategy": args.strategy, "n_robots": args.n_robots, "seed": args.seed,
                                   "name": args.name}.items() if v is not None}
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    if args.sweep:
        param, values = args.sweep
        vals = [_parse_value(param, v) for v in values.split(",")]
        df = run_sweep(cfg, param, vals, n_trials=args.trials, verbose=not args.quiet)
        print(aggregate(df, group_cols=["sweep_value"]).to_string())
    else:
        df = run_trials(cfg, n_trials=args.trials, verbose=not args.quiet, save_step_logs=not args.no_step_logs)
        print(aggregate(df).to_string())


if __name__ == "__main__":
    main()
