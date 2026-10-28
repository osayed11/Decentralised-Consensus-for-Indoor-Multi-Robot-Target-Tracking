# Multi-Robot Indoor Target Search & Tracking — Simulation

2D simulation of N robots searching for and tracking K relocating targets in an
indoor floor plan.  Built to the spec in `simulation-spec.md`.  The simulation
infrastructure is complete; the multi-agent-systems logic is **stubbed** and is
the research team's contribution.

## What is complete vs. stubbed

| Module | Status | Contents |
|---|---|---|
| `sim/environment.py` | complete | floor plan (default house/office, or from 2D array / image / .npy / .txt), occupancy grid, inflation, free-space sampling, Bresenham line-of-sight |
| `sim/robot.py` | complete | holonomic point-mass (default) or diff-drive kinematics, waypoint following, range + Gaussian-noise sensor (optional LOS), `Belief` container |
| `sim/target.py` | complete | random relocation (uniform / exponential / fixed / none), `pause_relocation()` / `resume_relocation()` |
| `sim/planning.py` | complete | A* on the inflated grid, 8-connected, string-pulled waypoints |
| `sim/faults.py` | complete | scheduled robot failure (stops sensing / moving / communicating) |
| `sim/dispatch.py` | complete (pipeline) | `query_target()` → select robot → pause relocation → plan to belief → drive → success / failure + logging |
| `sim/harness.py` | complete | `SimConfig` (YAML/JSON/dataclass), `Simulation` step loop, `run_trials`, `run_sweep`, CSV logs, mean/std/95% CI aggregation |
| `sim/visualize.py` | complete | 2D view (robots, sensor circles, true target, beliefs + covariance ellipses, comm edges), animation / frame export, result plots |
| **`sim/strategies.py`** | **STUB** | `CoordinationStrategy` interface + `IndependentStrategy`, `CentralizedStrategy`, `DecentralizedConsensusStrategy`. Placeholder: belief = last raw observation, no relocation detection, no messages |
| **`sim/comms.py`** | **STUB** | `CommunicationLayer` interface: `build_graph`, `send_message`, `broadcast`, message log / cost metrics. Placeholder: graph has no edges, messages are dropped |
| **`sim/dispatch.py::select_robot_for_dispatch`** | **STUB** | Placeholder: nearest alive robot by *true* distance (cheats on purpose so the pipeline runs) |

Everything marked `TODO (research team)` in the source is yours to replace.
Nothing outside those three places needs to change to drop in real algorithms.

## Install / run

```bash
pip install numpy matplotlib networkx pandas pyyaml   # pillow or ffmpeg for animation export
python run_experiments.py --config sim/configs/default.yaml --trials 10
python demo.py --duration 120                          # live animation
python demo.py --save results/demo.gif --duration 60   # export
```

Sweeps (each value becomes its own results folder plus a combined `sweep_<param>.csv`):

```bash
python run_experiments.py --config sim/configs/scalability.yaml --sweep n_robots 2,4,6,8 --trials 20
python run_experiments.py --config sim/configs/faults.yaml --sweep packet_loss_prob 0,0.2,0.5 --trials 20
python run_experiments.py --config sim/configs/default.yaml --sweep strategy independent,centralized,decentralized
```

## Per-step loop (`Simulation.step`)

1. apply scheduled robot failures
2. targets relocate if due (unless paused by a dispatch)
3. `comm.build_graph(robots, t)` **(stub)**
4. each robot senses; `strategy.update_belief(...)` and `strategy.detect_relocation(...)` **(stub)**; inbox is cleared
5. `strategy.get_messages_to_send(...)` → `comm.send_message` / `comm.broadcast` **(stub)**; delivered messages land in `robot.inbox` for the next step
6. scheduled queries → `Dispatcher.query_target` → `select_robot_for_dispatch` **(stub)**
7. active dispatches advance (success if within `dispatch_success_radius` of the true target; failure on arrival-without-target, timeout, or robot failure)
8. idle robots patrol to random free goals via A*; all robots step kinematics
9. log

## Outputs (`results/<name>/`)

* `trial_XXX_steps.csv` — per (t, target, robot): true pos, robot pos, belief, error, observed flag, cumulative messages
* `trial_XXX_events.csv` — relocations, relocation detections, queries, dispatch outcomes, robot failures
* `trial_XXX_dispatch.csv` — one row per query: robot, start/end time, success, reason, belief error at dispatch, distance
* `trials.csv` — one row per trial: `rmse_team`, `frac_time_localised`, `time_to_first_localisation`, `messages_sent`, `bytes_sent`, `dispatch_success_rate`, `dispatch_time_mean`, …
* `summary.csv` — mean / std / 95% CI / n per metric
* `config.json`

"Team" error = min over robots of belief error at each time (best-informed robot).

## Plotting

```python
import pandas as pd, matplotlib.pyplot as plt
from sim.visualize import plot_error_over_time, plot_rmse_over_time, plot_metric_bars, plot_degradation
plot_error_over_time(pd.read_csv("results/default/trial_000_steps.csv"), events_df=pd.read_csv("results/default/trial_000_events.csv"))
plot_rmse_over_time("results/default")
plot_metric_bars(pd.read_csv("results/default/sweep_strategy.csv"), "time_to_first_localisation", group="sweep_value")
plot_degradation(pd.read_csv("results/faults/sweep_packet_loss_prob.csv"), "rmse_team", hue=None)
plt.show()
```

## Config keys

See `SimConfig` in `sim/harness.py`; every field can be set from YAML/JSON.
Key ones: `n_robots`, `n_targets`, `strategy`, `strategy_kwargs`, `sensor_range`,
`comm_range`, `sensor_noise_std`, `use_line_of_sight`, `kinematics`
(`holonomic` | `diffdrive`), `relocation_dist/min/max/mean`, `packet_loss_prob`,
`robot_failures` (`{robot_id: t}`), `query_times` / `query_interval`,
`dispatch_success_radius`, `dispatch_timeout`, `dt`, `time_limit`, `n_trials`, `seed`.

## Implementing the research logic

* **Strategies** (`sim/strategies.py`): override `update_belief`, `detect_relocation`,
  `get_messages_to_send`. `neighbor_messages` is the list of payload dicts delivered
  to `robot.inbox` since the previous step. Return a `Belief(mean, cov, timestamp, source)`.
* **Comms** (`sim/comms.py`): `build_graph` should add an edge when
  `distance <= comm_range` between alive robots; `send_message` should check the edge,
  apply `packet_loss_prob` with `self.rng`, append a `MessageRecord` to `self.log`
  and push the payload into `receiver.inbox` on delivery.
* **Dispatch selection** (`sim/dispatch.py`): `select_robot_for_dispatch(robots, beliefs, target, env)`
  receives `beliefs[robot_id][target_id]`; it must not use `target.position` once the real version exists.
