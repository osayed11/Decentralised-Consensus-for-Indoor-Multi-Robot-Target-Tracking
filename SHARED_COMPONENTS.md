# Shared components: comms layer, belief, message format, estimator

Branch `shared-comms-estimator`. These are the parts every method uses. All three
strategies now run on them; what is still open is listed at the end of section 1.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib networkx pandas pyyaml pytest
pytest -q tests                      # run tests for the current branch (count varies)
python demo.py --strategy decentralized --duration 60   # comm edges + message count now visible
```

## 1. Status

| piece | file | status |
|---|---|---|
| Comms layer | `sim/comms.py` | **done**: range-based graph, packet loss, delivery to `robot.inbox`, logging |
| Belief contents | `sim/robot.py` (`Belief`) | **done**: added `last_seen`, `reset_count` (defaults keep old code working) |
| Message formats | `sim/messages.py` | **done**: `belief` and `observation` payloads + helpers |
| Shared estimator | `sim/estimator.py` | **done**: Kalman filter, innovation gating, belief reset, gating log |
| Strategies use it | `sim/strategies.py` | **done** for the shared part; `independent` is complete |
| Decentralized consensus update | `sim/strategies.py` | **TODO**: messages are already broadcast every step, but not yet fused |
| Centralized fusion at the hub | `sim/strategies.py` | **TODO** |
| Gating metrics (false resets / missed relocations) | harness | **TODO** (being built in parallel); the data it needs is logged, see 5.3 |
| Dispatch selection | `sim/dispatch.py` | unchanged (still the true-distance stub) |
| Independent exploration and tracking (separate development branch) | `sim/exploration.py`, `sim/tracking.py`, `sim/harness.py` | **implemented in independent MVP**, not automatically merged into `shared-comms-estimator`; see §10 |
| Shared local robot collision avoidance (separate development branch) | `sim/safety.py`, `sim/robot.py`, `sim/harness.py` | **implemented in independent MVP**; integration across strategies should be confirmed after merging; see §10 |

## 2. Communication layer (`sim/comms.py`)

* **Link**: edge (i, j) iff both robots are alive and `‖p_i − p_j‖ ≤ min(comm_range_i, comm_range_j)`.
  Rebuilt every step by the simulation (`build_graph`). No line-of-sight check for radio.
* **Send**: `send_message(sender, receiver, payload)` delivers a *deep copy* of the payload into
  `receiver.inbox` iff the edge exists and the packet is not lost (probability `packet_loss_prob`).
  The strategy reads it on the **next** step as `neighbor_messages`.
* **Broadcast**: `broadcast(sender, robots, payload)` = one `send_message` per current neighbour
  (so a broadcast to 2 neighbours costs 2 messages; see open decision D3).
* **Log**: every attempt is a `MessageRecord(t, sender, receiver, payload_size, delivered, reason)`
  with `reason ∈ {"delivered", "lost", "no_link"}`. Metrics: `messages_sent()` (attempts),
  `messages_delivered()`, `messages_lost()`, `bytes_sent()`.
* **Fairness — important for every comparison**: packet-loss draws come from the comms layer's own
  random stream (seeded once from the simulation RNG). Without this, a strategy that sends more
  messages would consume more random numbers and get *different target relocations and patrol
  routes*. Test: `test_every_strategy_sees_the_same_world`.

## 3. Belief (`sim/robot.py`)

| field | meaning |
|---|---|
| `mean` (2,) | estimated target position [m] |
| `cov` (2,2) | covariance [m²] |
| `timestamp` | last time the belief was predicted/updated |
| `last_seen` | time of the newest **real observation** behind the belief (own, or a neighbour's) |
| `reset_count` | how many relocation resets this robot's belief has had (local counter, diagnostics) |
| `source` | tag: `own_obs`, `reset`, `msg`, `fused`, ... |

## 4. Message formats (`sim/messages.py`)

Every payload: `type`, `sender`, `target_id`, `t_sent`.

| type | used by | extra fields | helper |
|---|---|---|---|
| `belief` | decentralized (neighbour ↔ neighbour); centralized (hub → robots) | `mean`, `cov`, `last_seen`, `reset_count` | `belief_message(sender, target, belief, t)` |
| `observation` | centralized (robot → hub) | `z`, `noise_std`, `t_obs` | `observation_message(sender, target, z, noise_std, t)` |

* An inbox holds messages for **all targets and types**: filter with `for_target(msgs, target_id, kind)`.
* **Whose information is newer** is decided by `last_seen`: `is_newer(msg, my_belief)` is true when the
  neighbour's belief rests on a later sighting. After a relocation, a neighbour that saw the target
  before the move must not pull my fresh belief back. `reset_count` is per robot, so it is **not**
  comparable across robots.
* `message_to_belief(msg)` turns a received `belief` message into a `Belief`.

## 5. Shared estimator (`sim/estimator.py`)

### 5.1 Model
State = target position (static between relocations).

* **Predict** to time t: `P ← P + q·Δt·I` (`process_noise` q). Not motion: it makes old beliefs less
  certain, so fresh information wins.
* **Update** with observation z, `R = σ²I` (σ = `robot.sensor_noise_std`):
  `ν = z − μ`, `S = P + R`, `d² = νᵀS⁻¹ν`.
  * `d² ≤ gate` → Kalman update (noise averaged away).
  * `d² > gate` → outlier. After `confirm` consecutive outliers → **reset**: `μ = z, P = R`,
    `reset_count += 1`, relocation declared. Outliers before that are not fused.
  * No prior belief → initialise from z (not counted as a reset).
* `gate` is a χ²(2 dof) threshold: `chi2_threshold(p) = −2 ln(1 − p)` (p = 0.99 → 9.21, 0.95 → 5.99).

### 5.2 Settings (from the config)
```yaml
strategy: decentralized
strategy_kwargs:
  gate_prob: 0.99        # or gate: 9.21 (overrides gate_prob)
  process_noise: 0.01    # m^2/s
  confirm: 1             # consecutive outliers before a reset
```
Sweep e.g. `--sweep strategy_kwargs...` or loop over configs; every strategy accepts these.

### 5.3 What the gating metrics can use
* `sim.strategy.estimator.decisions`: one `GateDecision(t, robot_id, target_id, d2, gate, outlier, reset)`
  per observation-update.
* Harness events: `relocation` (ground truth) and `relocation_detected` (a reset), both already logged.
  False reset = a reset with no true relocation since the robot last saw the target; missed relocation
  = a true relocation followed by an observation that was fused instead of resetting.

### 5.4 Using it inside a strategy
`_SharedEstimatorBase.update_belief` already does predict + gate + update/reset on the robot's own
observation. A strategy extends it, e.g. the decentralized TODO:

```python
def update_belief(self, robot, target_id, observation, neighbor_messages, t):
    b = super().update_belief(robot, target_id, observation, neighbor_messages, t)   # own observation
    for msg in messages.for_target(neighbor_messages, target_id):
        other = messages.message_to_belief(msg)
        # consensus step goes here: e.g. gate other.mean against b (Estimator.innovation),
        # adopt it if is_newer(msg, b) and it disagrees (relocation), otherwise fuse
        # (covariance intersection / weighted average)
    return b
```

## 6. Sanity check (same world for every method)

`default.yaml`, **queries off** (otherwise dispatch makes the worlds diverge, see 7), 30 paired trials:

| belief update | team RMSE | error while the target is seen | relocation detections (true: 4.6) |
|---|---|---|---|
| old stub (last observation) | 4.28 m | 0.376 m | 0 |
| estimator, gate 0.99 | 4.32 m | **0.120 m** | 14.6 |
| estimator, gate 0.999 | 4.32 m | 0.102 m | 5.3 |
| estimator, gate 0.99, confirm 3 | 4.35 m | 0.185 m | 4.8 |

* The filter is ~3× more accurate whenever a robot sees the target.
* Team RMSE barely moves: it is dominated by the periods when **no** robot sees the target after a
  relocation and everyone keeps a stale position. The estimator cannot fix that; information sharing
  (centralized / decentralized) is what has to.
* At gate 0.99 a robot observing at 10 Hz takes noise for a relocation about once per 100
  observations → 3× too many detections. This is the false-reset side of the planned gate sweep.

## 7. Notes for fair experiments
* Use the same seeds for every method (paired comparison) and report differences with CIs.
* Belief-driven behaviour (dispatch: the robot drives to *its* belief; relocation pauses while it does)
  makes worlds diverge between methods after the first query. For estimator-only questions run with
  `query_interval: null`; for the full system accept it and use more trials.

## 8. Decisions for the team to confirm
* **D1** Belief fields as in section 3 (is `reset_count` useful, or should it be a global epoch?).
* **D2** Message fields as in section 4.
* **D3** Cost model: broadcast = one message per neighbour (current) or one transmission?
* **D4** Decentralized robots broadcast every step (10 Hz, ~8,600 messages per 300 s trial with 3
  robots). A lower rate (e.g. 1 Hz) is a cheap parameter if communication cost matters.
* **D5** Defaults `gate_prob 0.99, process_noise 0.01, confirm 1`.
* **D6 — DECIDED (2026-11-22): off-board coordinator (base station).** Every alive robot can always
  reach it, regardless of comm range; the same `packet_loss_prob` applies in both directions. It stays
  up by default (robot failures only remove that robot's sightings), but **can fail via a scheduled
  fault** (`coordinator_failure_time`, default off): from then on it neither receives nor broadcasts, and
  centralised robots fall back to their local estimator (independent-like). Server failure is a
  separate robustness scenario, not part of the fault-free / packet-loss / robot-failure runs. Uplink and downlink messages both count
  toward communication cost. Report framing: the centralised baseline assumes building-wide
  infrastructure and is the upper bound; the decentralised method uses only short-range links.
  **Built:** `comms.send_to_coordinator` (robot -> server), `comms.coordinator_broadcast` (server ->
  all alive robots), receiver id `comms.COORDINATOR` in `get_messages_to_send`, and the strategy hook
  `coordinator_step(inbox, comm, robots, t)` that runs on the server each step. A robot-0 hub may be
  added later as an optional "realistic centralised" variant.
* **Failures — DECIDED (2026-11-22): unexpected failures.** Per-robot and per-server probability of
  failing during the trial, constant risk; permanent by default, optional downtime (a returning robot
  keeps its stale belief unless `keep_belief_on_recovery: false`); fixed-time failures kept for
  controlled runs. Failure times are drawn up front from their own random stream, and targets / each
  robot's patrol and sensing have separate streams too, so failures never change the rest of the
  world. Impact is measured around each failure (10 s before vs 30 s after, recovery time) rather than
  only over the whole trial. See `sim/faults.py`, README "Failures".
* **No-sighting rule — DECIDED (2026-11-22): forget the position, remember the check.** When a
  robot's believed position is within 80% of its sensor range (and visible, if line-of-sight sensing is
  on) and the target is not seen, the belief is cleared: the target teleports to a random spot, so the
  old position carries no information. The check is kept in the belief (`last_absence`, `absence_pos`,
  `absence_radius` = the area seen empty); `last_seen` is not changed. The next sighting starts a fresh
  belief and is not counted as a relocation reset. `messages.belief_contradicted(claim_pos,
  claim_last_seen, my_belief)` says whether a neighbour's claim is stale (older than my check and inside
  the area I saw empty), so a rumour cannot re-infect a robot. Decentralised robots broadcast an
  `absence` message instead of a belief while theirs is cleared; centralised robots should send it to
  the server. Settings in `strategy_kwargs`: `absence` (on/off), `absence_view_fraction`. Event:
  `absence_detected`. (Briefly implemented first as "keep the mean, raise the uncertainty"; switched
  because the old position is meaningless and kept producing fake resets.)
* **Metric — DECIDED (2026-11-22): accuracy is the mean over alive robots** (no belief = building
  centre, failed robots excluded); search efficiency = first time every alive robot is within 1 m.
  Implemented in `sim/metrics.py`, see README "Metrics".

## 9. Files changed
`sim/comms.py`, `sim/robot.py` (Belief fields), `sim/strategies.py` (shared base, decentralized
broadcast), new `sim/estimator.py`, `sim/messages.py`, `tests/`.

## 10. Independent exploration, safety and tracking MVP (separate development work)

> **Integration status:** This section documents the latest `independent_tracking_mvp` implementation, which builds on the shared estimator and communication code. It is **not a claim that these changes are merged into branch `shared-comms-estimator`**. When merging, reconcile changes to `sim/harness.py`, `sim/robot.py`, `sim/visualize.py`, configuration and tests. The shared estimator/comms specifications in §§2–9 remain relevant.

### 10.1 Scope and behaviour

The independent strategy now has a minimal working **Explore → Track → Reacquire** navigation policy:

| Mode | Trigger | Behaviour |
|---|---|---|
| **Explore** | No current local target observation or reacquisition has ended | Sample a random *reachable* free-space goal and follow an A* path; do not use target truth, peers' beliefs or coverage history to select the goal. |
| **Track** | **This robot** observes target 0 | Follow **its own** Kalman-filtered `robot.beliefs[0].mean` using A*. Stop approaching within `tracking_stop_radius`; replan if the estimate moves far enough or the replan interval elapses. |
| **Reacquire** | Target observation disappears after tracking | Investigate the last locally estimated destination; immediately resume tracking on a new sighting; otherwise return to exploration on goal completion/path exhaustion or after `reacquire_timeout`. |

The follower is intentionally minimal: **one target** (`n_targets: 1`) and the **independent** strategy. The feature is opt-in (`independent_tracking: false` by default in the general configuration, `true` in the example below). All movement still uses the existing A* planner and robot dynamics, including local safety checks. No inter-robot coordination messages are sent by the independent strategy. The simulation uses target ground truth for synthesising observations and scoring results, **not** as a tracking destination.

### 10.2 Robust random patrol (Phase 1)

`sim/exploration.py` provides an `IndependentPatrol` controller. On a new goal request it:

1. Samples destinations in the largest reachable free-space component.
2. Rejects candidates closer than `patrol_min_goal_distance`.
3. Verifies an A* route, with at most `patrol_max_goal_attempts` attempts.
4. Follows the route using the existing navigation/motion controller.
5. Drops stalled routes when progress over `patrol_stall_timeout` falls below `patrol_progress_distance`, then chooses a replacement.

Failed robots do not move or sense. A recovered robot resumes patrol with a new goal. Goal decisions depend on the robot's own state and the **known static map**, not other robots' live locations or target truth.

### 10.3 Private exploration memory (Phase 2)

`ExplorationMemory` maintains a separate `last_seen[row, col]` grid **for each robot**. `-inf` means the cell has not been observed; otherwise the value is the latest simulated observation time. Every `exploration_observe_interval`, the robot marks eligible reachable, clearance-safe grid-cell centres inside `sensor_range`. When `use_line_of_sight: true`, a cell also must pass `Environment.line_of_sight` so walls block visibility. Only live robots update their histories.

Memory is retained after temporary failures if `keep_belief_on_recovery: true`, otherwise reset. Each robot's coverage map remains private; the **union of all maps is computed only by offline evaluation**, never used as a control input.

**Important:** coverage history is still **passive**. Destination selection is random, not based on freshness or coverage. Coverage-aware waypoint scoring and multi-robot Voronoi/Lloyd control have **not** been implemented. The Week 4 lab informs the grid-based coverage concept, not the current robot control algorithm.

### 10.4 Local robot-to-robot collision avoidance

`sim/safety.py` and the motion integration in `sim/robot.py` / `sim/harness.py` add a **shared geometric safety layer**:

- The simulator provides nearby robot centres inside `robot_detection_range` (default **2.0 m**) as local proximity readings; this is **not radio communication** and exposes no neighbours' estimates or future paths.
- Each planned motion segment is clipped before entering a nearby robot's exclusion disc. The clearance is the two robot radii plus `robot_safety_margin`: with 0.30 m radii and 0.05 m margin, centre separation must stay at least **0.65 m**.
- The existing route is retained so the robot can retry movement next step; failed robots remain physical obstacles. Patrols can abandon paths after the existing stall timeout.
- This is **stop-and-wait, short-step collision prevention**, **not** a control barrier function, reciprocal collision-avoidance planner or guaranteed deadlock-free controller. Opposing robots may still block one another in narrow corridors. The proximity model has no occlusion/noise/missed detections and uses sequential short movement steps.

Use the same safety settings for all three strategies in controlled experiments. Changing the environment physically (e.g. a robot failing and blocking a doorway) may legitimately change another robot's trajectory even without communication.

### 10.5 Key configuration

The standalone MVP demonstration lives at `sim/configs/independent_exploration.yaml`:

```yaml
name: independent_exploration
strategy: independent
n_robots: 3
n_targets: 1
patrol: true
independent_tracking: true
tracking_stop_radius: 0.8
tracking_replan_distance: 0.5
tracking_replan_interval: 1.0
reacquire_timeout: 20.0
robot_avoidance: true
robot_detection_range: 2.0
robot_safety_margin: 0.05
patrol_min_goal_distance: 1.0
patrol_max_goal_attempts: 20
patrol_stall_timeout: 8.0
patrol_progress_distance: 0.15
exploration_observe_interval: 0.5
use_line_of_sight: true
query_interval: null
relocation_dist: uniform
relocation_min: 30.0
relocation_max: 90.0
dt: 0.1
time_limit: 120.0
n_trials: 3
seed: 0
output_dir: results
```

`query_interval: null` intentionally disables the separate dispatch mechanism: `sim/dispatch.py` still contains a ground-truth-based selection placeholder. `use_line_of_sight: true` is set **in this example**, while older configs may still default to `false`; match it in comparisons.

### 10.6 Run and inspect the independent MVP

From the **independent MVP repository root** (containing `demo.py`):

```bash
python -m pip install numpy pandas matplotlib networkx pyyaml pytest pillow
python -m pytest -q tests

# Animated demo with mode labels (explore / track / reacquire):
python demo.py --config sim/configs/independent_exploration.yaml --duration 60 --paths

# Headless animated output:
python demo.py --config sim/configs/independent_exploration.yaml --duration 30 --paths --save results/independent_tracking.gif

# Three seeded trials:
python run_experiments.py --config sim/configs/independent_exploration.yaml --trials 3

# Separate per-robot coverage visualisations:
python show_exploration.py --duration 60 --output results/coverage.png
```

The previously completed independent MVP development run reported **61 passing tests** (including tests for patrol, private memory, geometric safety and navigation-mode transitions). That count is **not** a test claim for the separate shared branch after integration.

The demo labels robot navigation modes and hides **unused radio graph lines** under the independent strategy; this is a visual choice, not a change to the range-based communication infrastructure described in §2. Check experiment logs for `navigation_mode`, state changes, `tracking_engagements` and `messages_sent`. Patrol/coverage diagnostics include `exploration_coverage`, `exploration_coverage_mean`, `exploration_coverage_union`, `patrol_goals_assigned`, `patrol_goals_rejected` and `patrol_stalls`; safety diagnostics include `robot_safety_stops` and `min_robot_separation`.

### 10.7 Files added or changed in the independent MVP

| File | Responsibility |
|---|---|
| `sim/exploration.py` (**new**) | Robust random patrol and private per-robot observation-time maps |
| `sim/tracking.py` (**new**) | Explore/Track/Reacquire state machine driven by own observations and Kalman belief |
| `sim/safety.py` (**new**) | Short swept-segment clearance computation against locally detected robots |
| `sim/harness.py` | Integrates patrol, sensing-memory updates, target-following, proximity input, mode/coverage/safety logging |
| `sim/robot.py` | Applies optional nearby-robot clearance during movement; counts safety-limited steps |
| `sim/visualize.py` | Displays navigation modes; suppresses misleading radio lines for independent demos |
| `sim/configs/independent_exploration.yaml` (**new**) | Working independent MVP scenario |
| `show_exploration.py` (**new**) | Plots per-robot exploration maps |
| `tests/test_exploration.py`, `tests/test_robot_safety.py`, `tests/test_tracking.py` (**new**) | Tests of added features |
| `tests/test_failures.py` | Updates a trajectory-equality assumption for runs with physical robot avoidance enabled |

### 10.8 Fairness, integration and outstanding work

- **Preserve the shared estimator/message definitions** in §§2–5. Independent tracking should **not** bypass the local Kalman filter or use other robots' information.
- **Use identical sensing, A*, motion, safety and exploration/tracking policies across all three strategies** when comparing information sharing. The MVP follower currently applies to the independent method only; do not interpret results as an information-sharing comparison until the common navigation layer is integrated across strategies.
- **Paired trial world versus behaviour:** matched seeds and dedicated RNG streams help preserve exogenous relocation/sensing randomness, but robots may physically take different routes after one strategy receives new information. That is an intended behavioural effect, not necessarily a broken pairing.
- **Metrics caveat:** existing `error_eval` substitutes the building centre when a robot has no belief. Report estimates conditional on `has_belief` separately from whole-team penalty metrics, and distinguish first *physical detection* from all-robot localisation accuracy.
- **Still open:** decentralised consensus fusion; centralised coordinator fusion; false-reset/missed-relocation metrics; removing the true-distance dispatch stub; coverage-aware goal selection; formal deadlock-free multi-robot traffic management. The latter two are optional improvements, not requirements for the minimal independent baseline.

### 10.9 Additional development notes

For more detail in the independent MVP snapshot, see `EXPLORATION_PHASES_1_2.md`, `LOCAL_ROBOT_AVOIDANCE.md` and `INDEPENDENT_TRACKING_MVP.md`.
