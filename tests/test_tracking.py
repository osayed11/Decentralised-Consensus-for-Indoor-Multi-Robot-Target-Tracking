"""Behaviour tests for the minimal independent follower."""
import numpy as np
import pytest

from sim.environment import Environment
from sim.harness import SimConfig, Simulation
from sim.robot import Belief, Robot
from sim.tracking import LocalTargetFollower


def test_follows_only_own_estimate_and_abandons_patrol():
    env = Environment.from_size(12, 12, resolution=0.5)
    robot = Robot(id=0, position=np.array([2.0, 2.0]))
    robot.set_path([np.array([2.0, 8.0])])
    follower = LocalTargetFollower(env)
    belief = Belief(mean=np.array([8.0, 2.0]), cov=np.eye(2))
    assert follower.step(robot, belief, np.array([8.1, 2.1]), 0.0) == 'track'
    np.testing.assert_allclose(robot.path[-1], belief.mean)
    assert follower.active


def test_no_detection_never_chases_stale_belief_from_exploration():
    env = Environment.from_size(12, 12, resolution=0.5)
    robot = Robot(id=0, position=np.array([2.0, 2.0]))
    follower = LocalTargetFollower(env)
    belief = Belief(mean=np.array([8.0, 2.0]), cov=np.eye(2))
    assert follower.step(robot, belief, None, 5.0) == 'explore'
    assert not robot.has_path


def test_loss_reacquire_then_timeout_returns_to_patrol():
    env = Environment.from_size(12, 12, resolution=0.5)
    robot = Robot(id=0, position=np.array([2.0, 2.0]))
    belief = Belief(mean=np.array([8.0, 2.0]), cov=np.eye(2))
    follower = LocalTargetFollower(env, reacquire_timeout=3.0)
    assert follower.step(robot, belief, belief.mean, 1.0) == 'track'
    assert follower.step(robot, belief, None, 2.0) == 'reacquire'
    assert follower.active
    assert follower.step(robot, belief, None, 5.0) == 'explore'
    assert not robot.has_path


def test_close_observation_stops_and_loss_resumes_patrol():
    env = Environment.from_size(12, 12, resolution=0.5)
    robot = Robot(id=0, position=np.array([2.0, 2.0]))
    belief = Belief(mean=np.array([2.2, 2.1]), cov=np.eye(2))
    follower = LocalTargetFollower(env)
    assert follower.step(robot, belief, belief.mean, 0.0) == 'track'
    assert not robot.has_path
    assert follower.step(robot, belief, None, 1.0) == 'explore'


def test_mvp_config_requires_independent_single_target():
    with pytest.raises(ValueError):
        SimConfig.from_dict({'independent_tracking': True, 'strategy': 'centralized'})
    with pytest.raises(ValueError):
        SimConfig.from_dict({'independent_tracking': True, 'n_targets': 2})


def test_end_to_end_independent_tracking_and_zero_messages():
    cfg = SimConfig.from_file('sim/configs/independent_exploration.yaml').with_overrides(
        time_limit=35.0, seed=2)
    sim = Simulation(cfg)
    sim.run()
    modes = set(sim.step_dataframe()['navigation_mode'])
    assert 'explore' in modes
    assert 'track' in modes
    assert sim.comm.messages_sent() == 0
    assert all(f.state.mode in {'explore', 'track', 'reacquire'} for f in sim.followers.values())
    assert 'tracking_engagements' in sim.summary()


def test_robots_do_not_follow_another_robots_detection():
    cfg = SimConfig.from_file('sim/configs/independent_exploration.yaml').with_overrides(
        n_robots=2, time_limit=0.5)
    sim = Simulation(cfg)
    first, second = sim.robots
    # Inject one local measurement into the independent navigation controller:
    # only the robot that received it may leave exploration.
    obs = np.array([5.0, 5.0])
    belief = Belief(mean=obs.copy(), cov=np.eye(2))
    assert sim.followers[first.id].step(first, belief, obs, 0.0) == 'track'
    assert sim.followers[second.id].step(second, Belief(), None, 0.0) == 'explore'
