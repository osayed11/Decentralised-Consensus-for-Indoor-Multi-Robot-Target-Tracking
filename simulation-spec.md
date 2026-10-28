# Multi-Robot Indoor Target Search & Tracking — Simulation Build Spec

## Scope boundary

**Build fully:** environment, robot physics/kinematics, sensor simulation, communication graph mechanics, path planning, fault injection, experiment harness, logging, visualization.

**Do NOT implement — stub only:** target belief fusion/update logic (independent / centralized / decentralized consensus), innovation-gating/relocation-detection logic, and dispatch-selection logic. These are the research team's actual contribution. Provide clean interfaces with placeholder implementations (e.g. "belief = last raw observation") so the simulation runs end-to-end and can be visually sanity-checked before real algorithms are dropped in.

---

## 1. Environment
- 2D space with a predefined floor plan: load from a simple format (2D array or image mask) representing free space vs. obstacles. Provide one default example layout (a simple house/office with a few rooms and corridors).
- Occupancy grid for path planning, configurable resolution.
- Configurable overall size (metres) and boundary walls.

## 2. Robots
- N robots (default 3, configurable up to 8+ for scalability runs).
- Each robot: position, heading, max speed, sensor range, communication range.
- Simple kinematics — recommend a holonomic point-mass model (moves directly toward a waypoint) for time efficiency; differential-drive as an optional stretch.
- Random collision-free initial positions.
- Waypoint-following controller that consumes a path from the planner.

## 3. Target(s)
- K targets (default 1), true (x, y) position.
- Random initial position in free space.
- Relocation: teleports to a new random free-space position at random intervals (configurable — e.g. uniform or Poisson-distributed interval).
- Expose `pause_relocation()` / `resume_relocation()` so the dispatch logic (built by the research team) can freeze movement during an active dispatch.

## 4. Sensor model (simulated — no real perception)
- Each timestep, per robot: if `distance(robot, target) <= sensor_range`, generate `observation = true_position + Gaussian_noise(std_dev)`. Out of range → `None`.
- Optional: line-of-sight raycast against obstacles for realism (flag as a later add-on, not required for v1).
- This is the full extent of "detection" — output feeds directly into the (stubbed) belief-update function.

## 5. Communication graph
- Each timestep: edge between robots i, j if `distance(i, j) <= comm_range`. Expose as an adjacency structure (networkx graph is fine).
- Generic message-passing API: `send_message(from, to, payload)` / `broadcast(from, payload)`, respecting the graph, logging every message for the communication-cost metric.
- Fault injection: configurable random packet-loss probability, and scheduled robot failure (robot stops sensing/moving/communicating from timestep t onward).

## 6. Path planning
- A* (or Dijkstra) on the occupancy grid: `plan_path(start, goal, grid) -> waypoints`.
- Reused both for general robot movement and for dispatch navigation.

## 7. Query / dispatch mechanism (infrastructure only)
- `query_target(target_id)` — simulates "user asks where the item is."
- Calls a pluggable `select_robot_for_dispatch(robots, beliefs) -> robot_id` — team-implemented; stub it as "nearest robot by true distance" so the pipeline runs before real logic exists.
- On dispatch: pause relocation, plan a path to the robot's current belief position, move it, log success/failure and time taken.

## 8. Experiment harness
- Config-driven (YAML/JSON or a Python dataclass): number of robots, strategy selection, trial count (default 100+), fault parameters, relocation interval, sensor noise, comm/sensor range, time limit per trial.
- Runs N trials per configuration with varying random seeds.
- Per-trial logging: true position over time, each robot's belief, error, messages sent, localisation success/failure, dispatch outcome.
- Output: structured CSV/pandas logs per trial, plus aggregated mean/std/confidence-interval summaries per configuration.

## 9. Visualization
- Matplotlib 2D view: floor plan, robots with sensor-range circles, true target, each robot's belief (with uncertainty ellipse if covariance is available), comm-graph edges.
- Animation/frame export for video demo use.
- Result-plotting utilities (RMSE over time, search-efficiency bars, robustness degradation curves) consuming the harness's logs.

## 10. The interface to fill in later (research team's contribution)

```python
class CoordinationStrategy:
    def update_belief(self, robot, observation, neighbor_messages):
        """Returns updated belief (position estimate + confidence/covariance).
        TO BE IMPLEMENTED: independent / centralized / decentralized logic."""
        raise NotImplementedError

    def get_messages_to_send(self, robot, comm_graph):
        """Returns messages this robot sends this timestep.
        TO BE IMPLEMENTED."""
        raise NotImplementedError
```

Provide three stub subclasses — `IndependentStrategy`, `CentralizedStrategy`, `DecentralizedConsensusStrategy` — each with a trivial placeholder (e.g. belief = most recent raw observation) so the full pipeline runs before any real algorithm is added.

---

## Suggested stack
Python 3.10+, numpy, matplotlib, networkx (optional — plain adjacency lists work too), pandas, PyYAML if config files are wanted. No external simulator (Gazebo/Isaac) needed given the 2D scope — keep dependencies minimal.

## Suggested file layout
```
/sim
  environment.py   # floor plan, obstacles, occupancy grid
  robot.py         # robot class, kinematics, sensor model
  target.py        # target class, relocation logic
  comms.py         # comm graph, message passing, fault injection
  planning.py      # A* path planning
  strategies.py    # CoordinationStrategy interface + stubs  <- RESEARCH TEAM FILLS IN
  dispatch.py      # query/dispatch logic, stub selection    <- RESEARCH TEAM FILLS IN
  harness.py       # experiment runner, config, logging
  visualize.py     # plotting/animation
  configs/         # example YAML configs
/results           # output logs, CSVs
README.md          # how to run, what's stubbed vs. complete
```

## Deliverable
A working end-to-end simulation (robots move and "track" using the stub logic) runnable via the experiment harness, with the three `CoordinationStrategy` subclasses clearly marked as placeholders ready for the research team's actual consensus/gating/dispatch algorithms to be dropped in without touching the surrounding infrastructure.
