# Multi-Robot Indoor Target Search & Tracking — Simulation

2D simulation of N robots searching for and tracking K relocating targets in an
indoor floor plan.  Built to the spec in `simulation-spec.md`.  The simulation
infrastructure is complete; the multi-agent-systems logic is **stubbed** and is
the research team's contribution.

## What is complete vs. stubbed

| Module | Status | Contents |
|---|---|---|
| `sim/environment.py` | complete | floor plan (default house/office, or from 2D array / image / .npy / .txt), occupancy grid, inflation, free-space sampling, Bresenham line-of-sight |
| `sim/robot.py` | complete | holonomic point-mass (default, with acceleration limit and corner lookahead) or diff-drive kinematics, waypoint following, range + Gaussian-noise sensor (optional LOS), `Belief` container |
| `sim/target.py` | complete | random relocation (uniform / exponential / fixed / none); carried (walked along a planned path at `carry_speed`) or teleport; `pause_relocation()` / `resume_relocation()` |
| `sim/planning.py` | complete | A* on the inflated grid, 8-connected, string-pulled waypoints |
| `sim/faults.py` | complete | unexpected robot and central-server failures: constant-risk probability or fixed time, permanent or temporary (see Failures) |
| `sim/dispatch.py` | complete (pipeline) | `query_target()` → select robot → pause relocation → plan to belief → drive → success / failure + logging |
| `sim/harness.py` | complete | `SimConfig` (YAML/JSON/dataclass), `Simulation` step loop, `run_trials`, `run_sweep`, CSV logs, mean/std/95% CI aggregation |
| `sim/estimator.py`, `sim/messages.py` | complete | shared resting/carried (IMM) filter, or the single static Kalman filter, with innovation gating and belief reset; agreed message formats |
| `sim/visualize.py` | complete | 2D view (robots, sensor circles, true target, beliefs + covariance ellipses, comm edges), animation / frame export, result plots |
| **`sim/strategies.py`** | **partly done** | All three strategies run the shared estimator (`sim/estimator.py`); `independent` is complete. **TODO**: centralized fusion at the hub, decentralized consensus update (messages are already broadcast). See `SHARED_COMPONENTS.md` |
| `sim/comms.py` | complete | range-based graph, packet loss (own random stream), delivery to `robot.inbox`, per-message log and cost metrics. See `SHARED_COMPONENTS.md` |
| **`sim/dispatch.py::select_robot_for_dispatch`** | **STUB** | Placeholder: nearest alive robot by *true* distance (cheats on purpose so the pipeline runs) |

Everything marked `TODO (research team)` in the source is yours to replace.
Nothing outside those three places needs to change to drop in real algorithms.

## Install / run

```bash
pip install numpy matplotlib networkx pandas pyyaml pytest   # pillow or ffmpeg for animation export
pytest -q tests                                              # shared components
python run_experiments.py --config sim/configs/default.yaml --trials 10
python demo.py --duration 120                          # live animation (2x real time; --speed 1 for real time)
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
* `trial_XXX_events.csv` — relocations (move starts; x, y = destination), arrivals, relocation detections, queries, dispatch outcomes, robot failures
* `trial_XXX_dispatch.csv` — one row per query: robot, start/end time, success, reason, belief error at dispatch, distance
* `trials.csv` — one row per trial: `rmse_mean`, `frac_robots_localised`, `rmse_worst`, `rmse_best`, `time_all_localised`, `search_success_all`, `time_any_localised`, `messages_sent`, `bytes_sent`, `dispatch_success_rate`, `dispatch_time_mean`, …
* `summary.csv` — mean / std / 95% CI / n per metric
* `config.json`

## Target motion (`sim/target.py`)

By default the target is **carried**: when a move is due, a person picks it up and
walks it to a new uniformly random reachable position along a collision-free path
(the same planner the robots use) at `carry_speed` (1.2 m/s, twice the robots' top
speed), puts it down, and it rests for a time drawn from `relocation_dist`
(measured from arrival). Robots can see it while it is in transit.
`target_motion: teleport` restores the instant jump (equivalent to an infinite carry
speed), so carry speed can be swept as an experiment variable.

* Events: `relocation` when a move starts (x, y = destination), `arrival` when it is put down.
* Step log: `target_moving` marks steps where the target is in transit.
* Dispatch pausing only delays the *start* of a move; a target already being carried keeps going.
* Moves draw only on the target random stream, so runs stay paired across strategies and faults.

## Failures (`sim/faults.py`)

Failures are *unexpected*: each robot fails during a trial with probability
`robot_failure_prob`, at a moment drawn with constant risk (nothing hints that
it is coming). The central server has its own `coordinator_failure_prob`
(only the centralised strategy uses a server). Failures are permanent unless
`robot_downtime` / `coordinator_downtime` is set; a returning robot keeps its
stale beliefs unless `keep_belief_on_recovery: false`. Fixed-time failures
(`robot_failures: {1: 100.0}`, `coordinator_failure_time`) override the draw,
for controlled scenarios. All failure times are drawn at the start of the
trial from their own random stream, and targets, each robot's patrol and each
robot's sensor noise also have their own streams, so the same seed gives the
same failures for every strategy and a faulty run differs from its fault-free
twin only by the failure itself. Example configs: `failures.yaml`, `server_failure.yaml`.

## Resting / carried estimator (`sim/estimator.py`)

The shared estimator assumes the target is either **resting** or **being carried**
and runs an Interacting Multiple Model (IMM) filter: one Kalman filter per mode on
a shared position-and-velocity state, plus the probability of each mode.

* Resting: velocity is zero; position uncertainty grows slowly (`process_noise`).
* Carried: constant velocity with acceleration noise `accel_noise` for turns and
  stops; on steps without a sighting the velocity decays over `velocity_decay_time`,
  so an unseen carry coasts a few metres instead of through walls.
* Mode switches follow the mean rest and walk durations (`mean_rest`, `mean_walk`).
* Uncertainty is capped at walking pace (`init_speed_std`) and the building's
  spread (`max_pos_std`).
* `Belief.mean` / `cov` are the blended position, so dispatch, messages and
  consensus code see the same interface; `Belief.velocity` and `Belief.p_moving`
  are extra. If code outside the estimator overwrites `mean` / `cov` (e.g. a
  consensus step), the next update re-centres both modes on the new values.
* Gating, reset and the no-sighting rule are unchanged, except that a sighting is
  an outlier only if neither mode explains it.
* `strategy_kwargs: {model: static}` restores the single static filter (ablation).
  The step log has a `p_moving` column.

## No-sighting rule (`sim/estimator.py`)

The sensor never misses a target that is in range, so when a robot's believed
position is well inside its view (80% of sensor range, and visible if
line-of-sight sensing is on) and the target is not there, the shared estimator
clears the belief: the target teleports to a random spot, so the old position
says nothing about the new one. The robot keeps a record of the check (when, and
the area it saw empty) so a neighbour's older belief pointing into that area is
recognised as stale. Decentralised robots share the check as an `absence`
message while they have no belief. A cleared robot is scored as guessing the
building centre (see Metrics). Switch off with `strategy_kwargs: {absence: false}`.
Trials report `n_absence_detections` next to `n_relocation_resets`.

## Metrics (accuracy and search efficiency, `sim/metrics.py`)

Each alive robot's own belief is scored against the true target position.
A robot with no belief yet counts as guessing the centre of the building; a
failed robot is excluded from the moment it fails. Per time step:

| metric | meaning | role |
|---|---|---|
| `rmse_mean` | RMSE over alive robots and time | headline accuracy |
| `frac_robots_localised` | average share of robots within `localised_threshold` (1 m) | companion |
| `rmse_worst` | RMSE of the worst robot each step | robustness view |
| `rmse_best` | RMSE of the best robot each step | "does anyone know" |
| `time_all_localised` / `search_success_all` | first time every alive robot is within 1 m | headline search efficiency |
| `time_any_localised` / `search_success_any` | first time any robot is within 1 m | secondary |
| `robot_failure_err_before` / `_after` / `_increase` | survivors' mean error in the 10 s before vs 30 s after each robot failure | failure impact |
| `robot_failure_recovery_time` / `_recovered` | time until survivors' error is back at the pre-failure level (share that recovered) | failure impact |
| `coordinator_failure_*` | the same around a server failure, over all alive robots | failure impact |

The step logs keep the raw `error` (NaN without a belief) and add `error_eval`
(the value the metrics use) and `has_belief`.

## Plotting

```python
import pandas as pd, matplotlib.pyplot as plt
from sim.visualize import plot_error_over_time, plot_rmse_over_time, plot_metric_bars, plot_degradation
plot_error_over_time(pd.read_csv("results/default/trial_000_steps.csv"), events_df=pd.read_csv("results/default/trial_000_events.csv"))
plot_rmse_over_time("results/default")
plot_metric_bars(pd.read_csv("results/default/sweep_strategy.csv"), "time_all_localised", group="sweep_value")
plot_degradation(pd.read_csv("results/faults/sweep_packet_loss_prob.csv"), "rmse_mean", hue=None)
plt.show()
```

## Config keys

See `SimConfig` in `sim/harness.py`; every field can be set from YAML/JSON.
Key ones: `n_robots`, `n_targets`, `strategy`, `strategy_kwargs`, `max_speed`, `max_accel`, `sensor_range`,
`comm_range`, `sensor_noise_std`, `use_line_of_sight`, `kinematics`, `robot_failure_prob`, `robot_downtime`, `keep_belief_on_recovery`, `coordinator_failure_prob`, `coordinator_failure_time`, `coordinator_downtime`,
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
