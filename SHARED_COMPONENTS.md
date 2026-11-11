# Shared components: comms layer, belief, message format, estimator

Branch `shared-comms-estimator`. These are the parts every method uses. All three
strategies now run on them; what is still open is listed at the end of section 1.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib networkx pandas pyyaml pytest
pytest -q tests                      # 20 tests, ~2 s
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
* **D6** Centralized hub: robot 0 over the same radio graph (hub out of range → no fusion), or an
  infrastructure link that always works?

## 9. Files changed
`sim/comms.py`, `sim/robot.py` (Belief fields), `sim/strategies.py` (shared base, decentralized
broadcast), new `sim/estimator.py`, `sim/messages.py`, `tests/`.
