"""Local robot collision avoidance, independent of the communication layer."""
import numpy as np

from sim.environment import Environment
from sim.harness import SimConfig, Simulation
from sim.robot import Robot
from sim.safety import safe_motion_fraction


def test_swept_segment_catches_tunnelling():
    start = np.array([1., 1.])
    delta = np.array([4., 0.])
    f = safe_motion_fraction(start, delta, [np.array([3., 1.])], .65)
    assert 0 < f < 1
    assert np.linalg.norm(start + f * delta - [3, 1]) >= .65 - 1e-7


def test_motion_away_from_touching_robot_allowed():
    assert safe_motion_fraction(np.array([1., 1.]), np.array([-1., 0.]),
                                [np.array([1.65, 1.])], .65) == 1
    assert safe_motion_fraction(np.array([1., 1.]), np.array([1., 0.]),
                                [np.array([1.65, 1.])], .65) == 0


def test_robot_stops_at_other_robot_not_inside_it():
    env = Environment.from_size(9, 7, resolution=.2, robot_radius=.3)
    r = Robot(0, np.array([2., 3.]), max_speed=8., max_accel=0.)
    r.set_path([np.array([6., 3.])])
    other = np.array([4., 3.])
    r.step(.5, env, local_robot_positions=[other], robot_clearance=.65)
    assert np.linalg.norm(r.position - other) >= .65 - 1e-6
    assert r.robot_safety_stops == 1
    assert r.has_path  # waits, then can retry or eventually replan


def test_two_approaching_robots_remain_separated():
    env = Environment.from_size(9, 7, resolution=.2, robot_radius=.3)
    a = Robot(0, np.array([2., 3.]), max_speed=.6, max_accel=0.)
    b = Robot(1, np.array([5., 3.]), max_speed=.6, max_accel=0.)
    a.set_path([np.array([6., 3.])])
    b.set_path([np.array([1., 3.])])
    for _ in range(70):
        a.step(.1, env, local_robot_positions=[b.position.copy()], robot_clearance=.65)
        b.step(.1, env, local_robot_positions=[a.position.copy()], robot_clearance=.65)
        assert np.linalg.norm(a.position - b.position) >= .65 - 1e-6
    assert a.robot_safety_stops + b.robot_safety_stops > 0


def test_disabled_avoidance_preserves_legacy_motion():
    env = Environment.from_size(9, 7, resolution=.2, robot_radius=.3)
    a = Robot(0, np.array([2., 3.]), max_speed=2., max_accel=0.)
    a.set_path([np.array([6., 3.])])
    a.step(.2, env)
    assert np.allclose(a.position, [2.4, 3.])
    assert a.robot_safety_stops == 0


def test_full_simulation_collision_free_and_no_messages():
    cfg = SimConfig(n_robots=5, time_limit=15., query_interval=None,
                    robot_avoidance=True, use_line_of_sight=True)
    sim = Simulation(cfg, seed=7)
    for _ in range(150):
        sim.step()
        for i, a in enumerate(sim.robots):
            for b in sim.robots[i + 1:]:
                assert np.linalg.norm(a.position-b.position) >= .65 - 1e-6
    assert sim.comm.messages_sent() == 0
