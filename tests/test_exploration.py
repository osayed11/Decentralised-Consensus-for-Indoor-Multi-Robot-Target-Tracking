import numpy as np

from sim.environment import Environment
from sim.exploration import ExplorationMemory, IndependentPatrol
from sim.harness import SimConfig, Simulation
from sim.robot import Robot


def test_private_memory_initially_unseen_and_finite_after_sensing():
    env = Environment.from_size(8, 8, resolution=0.5)
    mem = ExplorationMemory(env)
    assert mem.coverage_fraction == 0
    pos = np.array([3.5, 3.5])
    mem.observe(pos, 1.3, 2.0, env)
    assert 0 < mem.coverage_fraction < 1
    r, c = env.world_to_grid(pos)
    assert mem.last_seen[r, c] == 2.0
    assert np.isinf(mem.age(2.0)[0, 0])
    mem.observe(pos, 1.3, 4.0, env)
    assert mem.age(6.0)[r, c] == 2.0
    mem.clear()
    assert mem.coverage_fraction == 0


def test_memory_respects_sensor_range_and_occlusion():
    grid = np.zeros((20, 20), dtype=bool)
    grid[:, 10] = True
    env = Environment.from_array(grid, resolution=0.5, robot_radius=0)
    pos = np.array([4.25, 5.25])
    mem = ExplorationMemory(env)
    mem.observe(pos, 2.0, 1.0, env, use_line_of_sight=True)
    assert not mem.observed_mask[10, 11]  # behind wall, though within 2 m
    assert mem.observed_mask[10, 9]
    other = ExplorationMemory(env)
    assert other.coverage_fraction == 0


def test_patrol_selects_reachable_nontrivial_goal_and_is_reproducible():
    env = Environment.from_size(8, 8, resolution=0.5)
    p = np.array([2.0, 2.0])
    robots = [Robot(id=0, position=p.copy()) for _ in range(2)]
    patrols = [IndependentPatrol(env, np.random.default_rng(7)) for _ in range(2)]
    for ctrl, robot in zip(patrols, robots):
        ctrl.step(robot, 0)
        assert robot.has_path
        assert np.linalg.norm(ctrl.state.goal - p) >= 1.0
        assert env.is_reachable(ctrl.state.goal)
    np.testing.assert_allclose(patrols[0].state.goal, patrols[1].state.goal)


def test_stall_discards_old_path_and_replans():
    env = Environment.from_size(8, 8, resolution=0.5)
    r = Robot(id=0, position=np.array([2.0, 2.0]))
    ctrl = IndependentPatrol(env, np.random.default_rng(8), stall_timeout=1.0)
    ctrl.step(r, 0.0)
    assert r.has_path
    ctrl.step(r, 1.1)  # robot has not moved
    assert ctrl.state.stalls == 1
    assert ctrl.state.goals_assigned == 2


def test_sim_coverage_independent_and_failure_pauses_updates():
    cfg = SimConfig(seed=4, n_robots=2, time_limit=2, query_interval=None,
                    robot_failures={0: 0.5}, robot_downtime=0.5,
                    use_line_of_sight=True)
    sim = Simulation(cfg)
    for _ in range(7):
        sim.step()
    p0, p1 = sim.patrols[0], sim.patrols[1]
    assert p0.memory is not p1.memory
    assert p0.memory.coverage_fraction > 0
    assert p1.memory.coverage_fraction > 0
    assert np.max(p0.memory.last_seen[np.isfinite(p0.memory.last_seen)]) <= 0.5 + 1e-9
    sim.run()
    assert np.max(p0.memory.last_seen[np.isfinite(p0.memory.last_seen)]) > 1.0
    assert "exploration_coverage_union" in sim.summary()
    assert sim.comm.messages_sent() == 0


def test_no_recovery_memory_on_reboot():
    cfg = SimConfig(seed=7, n_robots=1, robot_failures={0: 0.6},
                    robot_downtime=0.5, keep_belief_on_recovery=False, query_interval=None)
    sim = Simulation(cfg)
    while sim.t < 0.6:
        sim.step()
    assert sim.patrols[0].memory.coverage_fraction > 0
    while sim.t < 1.1:
        sim.step()
    sim.step()
    assert np.isfinite(sim.patrols[0].memory.last_seen).any()
    assert np.max(sim.patrols[0].memory.last_seen[np.isfinite(sim.patrols[0].memory.last_seen)]) >= 1.1
