import numpy as np
import pytest

from sim.comms import COORDINATOR, CommunicationLayer
from sim.environment import Environment
from sim.harness import SimConfig, Simulation
from sim.messages import belief_message, observation_message, absence_message, message_to_belief
from sim.robot import Belief, Robot
from sim.strategies import CentralizedStrategy


def setup_strategy(**kwargs):
    robots = [Robot(i, np.array([2.0 + i, 2.0])) for i in range(3)]
    strategy = CentralizedStrategy(**kwargs)
    strategy.reset(robots, np.random.default_rng(0))
    return strategy, robots, CommunicationLayer()


def obs(sender, xy, t, target=0):
    return observation_message(sender, target, np.array(xy), 0.3, t)


def update(s, robot, observation, msgs, t):
    b = s.update_belief(robot, 0, observation, msgs, t)
    robot.beliefs[0] = b
    return b


def test_server_fuses_duplicates_once_and_preserves_time():
    s, robots, comm = setup_strategy(process_noise=0)
    msgs = [obs(0, [4, 4], 0), obs(2, [4.2, 4], 0)]
    b = message_to_belief(s.coordinator_step(msgs + msgs, comm, robots, 0.1)[0])
    np.testing.assert_allclose(b.mean, [4.1, 4])
    assert b.cov[0, 0] == pytest.approx(0.045001)
    assert b.last_seen == 0 and b.timestamp == 0.1
    again = message_to_belief(s.coordinator_step(msgs, comm, robots, 0.2)[0])
    np.testing.assert_allclose(again.cov, b.cov)


def test_batch_reset_is_logged_and_stale_data_ignored():
    s, robots, comm = setup_strategy()
    s.coordinator_step([obs(0, [2, 2], 0)], comm, robots, 0)
    s.coordinator_step([obs(1, [8, 8], 1), obs(2, [8.1, 8], 1)], comm, robots, 1)
    b = s._hub[0]
    assert b.reset_count == 1
    assert any(d.robot_id == COORDINATOR and d.reset for d in s.estimator.decisions)
    s.coordinator_step([obs(0, [2, 2], 0.5)], comm, robots, 1.2)
    np.testing.assert_allclose(s._hub[0].mean, b.mean)


def test_client_replaces_without_double_counting():
    s, robots, comm = setup_strategy(process_noise=0, absence=False)
    local = update(s, robots[1], np.array([4., 4.]), [], 0)
    hub = Belief(np.array([4.1, 4]), np.eye(2) * 0.045, timestamp=0.1, last_seen=0)
    msg = belief_message(COORDINATOR, 0, hub, 0.1)
    received = update(s, robots[1], None, [msg], 0.2)
    np.testing.assert_allclose(received.cov, hub.cov)
    again = update(s, robots[1], None, [msg], 0.3)
    np.testing.assert_allclose(again.cov, hub.cov)
    np.testing.assert_allclose(s._local[(1, 0)].cov, local.cov)
    fresh = update(s, robots[1], np.array([8., 8.]), [msg], 0.4)
    np.testing.assert_allclose(fresh.mean, [8, 8])


@pytest.mark.parametrize('condition,shared', [('normal', True), ('no_radio', True),
                                              ('lost', False), ('server_down', False),
                                              ('robot_zero_failed', True)])
def test_simulation_infrastructure_links(condition, shared):
    cfg = SimConfig(strategy='centralized', n_robots=3, n_targets=2, patrol=False,
                    relocation_dist='none', sensor_range=0.5, comm_range=0.1 if condition == 'no_radio' else 10,
                    packet_loss_prob=1 if condition == 'lost' else 0,
                    coordinator_failure_time=0 if condition == 'server_down' else None)
    sim = Simulation(cfg, env=Environment.from_size(12, 12))
    for r, pos in zip(sim.robots, ([2, 2], [5, 2], [8, 2])):
        r.position = np.array(pos, dtype=float)
    sim.targets[0].position = np.array([5., 2.])
    sim.targets[1].position = np.array([5., 2.2])
    if condition == 'robot_zero_failed':
        sim.robots[0].fail(0)
    sim.step()
    assert not sim.robots[2].beliefs[0].valid
    sim.step()
    for tid in (0, 1):
        assert sim.robots[2].beliefs[tid].valid == shared
        assert sim.robots[1].beliefs[tid].valid
    if shared:
        assert sim.robots[2].beliefs[0].last_seen == 0
        assert sim.robots[2].beliefs[0].source == 'msg'


def test_server_recovery_and_trial_reset():
    s, robots, comm = setup_strategy()
    comm.set_coordinator_up(False)
    assert s.coordinator_step([obs(0, [4, 4], 0)], comm, robots, 0) == []
    comm.set_coordinator_up(True)
    assert s.coordinator_step([obs(0, [4, 4], 1)], comm, robots, 1)
    s.reset(robots, np.random.default_rng(1))
    b = message_to_belief(s.coordinator_step([obs(1, [2, 2], 0)], comm, robots, 0)[0])
    np.testing.assert_allclose(b.mean, [2, 2])


def test_simultaneous_noiseless_observations():
    s, robots, comm = setup_strategy(process_noise=0)
    msgs = [observation_message(i, 0, np.array([4., 4.]), 0, 0) for i in (0, 1, 2)]
    b = message_to_belief(s.coordinator_step(msgs, comm, robots, 0)[0])
    np.testing.assert_allclose(b.mean, [4, 4])
    assert np.all(np.linalg.eigvalsh(b.cov) > 0)


def test_local_absence_of_server_estimate_propagates_and_blocks_stale_reply():
    s, robots, comm = setup_strategy()
    replies = s.coordinator_step([obs(0, [4, 2], 0)], comm, robots, 0)
    b = update(s, robots[2], None, replies, 0.1)
    assert not b.valid and s.detect_absence(robots[2], 0)
    outgoing = s.get_messages_to_send(robots[2], comm, robots, 0.1)
    assert outgoing[0][0] == COORDINATOR and outgoing[0][1]['type'] == 'absence'
    cleared = s.coordinator_step([outgoing[0][1]], comm, robots, 0.1)
    assert cleared[0]['type'] == 'absence'
    assert not s._hub[0].valid
    stale = dict(replies[0], t_sent=0.2)
    assert not update(s, robots[2], None, [stale], 0.2).valid
    assert s.coordinator_step([obs(1, [9, 9], 0.3)], comm, robots, 0.3)[0]['type'] == 'belief'


def test_server_absence_clears_client_but_preserves_newer_sighting():
    s, robots, comm = setup_strategy(absence=True)
    update(s, robots[0], np.array([8., 8.]), [], 0)
    checked = s.estimator.apply_absence(Belief(), 1, np.array([8., 8.]), 1)
    msg = absence_message(COORDINATOR, 0, checked, 1)
    assert not update(s, robots[0], None, [msg], 1.1).valid
    fresh = update(s, robots[0], np.array([8., 8.]), [msg], 1.2)
    assert fresh.valid and fresh.last_seen == 1.2
