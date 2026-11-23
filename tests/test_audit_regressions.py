import numpy as np

from sim.dispatch import Dispatcher
from sim.environment import Environment
from sim.estimator import Estimator, EstimatorConfig
from sim.robot import Belief, Robot
from sim.target import Target
from sim.strategies import CentralizedStrategy
from sim.messages import absence_message
from sim.harness import Simulation, SimConfig
from sim.dispatch import DispatchResult
from sim.metrics import accuracy_metrics
from sim.comms import CommunicationLayer
import pandas as pd


def dispatch_setup():
    env = Environment.from_size(12, 12)
    robot = Robot(0, np.array([2., 2.]))
    target = Target(0, np.array([9., 9.]))
    robot.beliefs[0] = Belief(np.array([4., 4.]))
    dispatcher = Dispatcher(env)
    dispatcher.query_target(target, [robot], 0)
    return dispatcher, robot, target


def test_successful_replan_is_not_a_completion():
    dispatcher, robot, target = dispatch_setup()
    robot.path = []
    robot.beliefs[0] = Belief(np.array([6., 6.]))
    assert dispatcher.step(1, [robot]) == []
    assert target.id in dispatcher.active and robot.has_path


def test_busy_robot_cannot_receive_second_assignment():
    dispatcher, robot, target = dispatch_setup()
    original = [p.copy() for p in robot.path]
    other = Target(1, np.array([8., 9.]))
    robot.beliefs[1] = Belief(np.array([7., 7.]))
    result = dispatcher.query_target(other, [robot], 1)
    assert result.reason == 'no_robot_available'
    assert list(dispatcher.active) == [target.id]
    np.testing.assert_allclose(robot.path, original)
    assert not other.paused


def test_failed_robot_near_target_cannot_succeed():
    dispatcher, robot, target = dispatch_setup()
    robot.position = target.position.copy()
    robot.fail(1)
    result, = dispatcher.step(1, [robot])
    assert result.success is False and result.reason == 'robot_failed'
    assert not target.paused


def test_local_noiseless_filter_does_not_become_singular():
    est = Estimator(EstimatorConfig(process_noise=0))
    b = est.update(0, 0, Belief(), np.zeros(2), 0, 0)
    b = est.update(0, 0, b, np.zeros(2), 0, 0.1)
    assert np.all(np.linalg.eigvalsh(b.cov) > 0)


def test_shared_absence_does_not_clear_a_position_behind_a_wall():
    grid = np.zeros((24, 24), dtype=bool)
    grid[:, 12] = True
    env = Environment(grid=grid, resolution=0.25, robot_radius=0)
    strategy = CentralizedStrategy()
    strategy.reset([], np.random.default_rng(0))
    strategy.bind(env, use_line_of_sight=True)
    b = Belief(np.array([3.5, 2.]), np.eye(2), timestamp=0, last_seen=0)
    empty = Estimator.apply_absence(Belief(), 1, np.array([2.5, 2.]), 2.4)
    msg = absence_message(0, 0, empty, 1)
    assert strategy._receive_absence(b, msg).valid


def test_radio_only_absence_still_clears_visible_old_estimate():
    strategy = CentralizedStrategy()
    b = Belief(np.array([3.5, 2.]), np.eye(2), timestamp=0, last_seen=0)
    empty = Estimator.apply_absence(Belief(), 1, np.array([2.5, 2.]), 2.4)
    assert not strategy._receive_absence(b, absence_message(0, 0, empty, 1)).valid


def test_dispatch_summary_exposes_censoring_without_redefining_rate():
    sim = Simulation(SimConfig(patrol=False))
    sim.step()
    sim.dispatcher.results = [DispatchResult(0, 0, 0, 0, t_end=1, success=True),
                              DispatchResult(1, 0, 0, 0, t_end=1, success=False),
                              DispatchResult(2, 0, 0, 0)]
    summary = sim.summary()
    assert summary['n_queries_issued'] == 3
    assert summary['n_dispatch_completed'] == 2
    assert summary['n_dispatch_unfinished'] == 1
    assert summary['n_dispatch_successful'] == summary['n_dispatch_failed'] == 1
    assert summary['dispatch_success_rate'] == 0.5


def test_existing_localisation_metric_can_succeed_without_beliefs():
    # Characterization, not an endorsement: preserve the agreed metric definition.
    df = pd.DataFrame([dict(t=0, target_id=0, robot_id=i, has_belief=False,
                            robot_failed=False, error_eval=0.2) for i in range(3)])
    result = accuracy_metrics(df)
    assert result['search_success_all'] == 1
    assert result['frac_robots_with_belief'] == 0


def test_radio_chain_does_not_automatically_forward_messages():
    robots = [Robot(i, np.array([float(i), 0]), comm_range=1.1) for i in range(3)]
    comm = CommunicationLayer()
    comm.build_graph(robots, 0)
    assert comm.graph.has_edge(0, 1) and comm.graph.has_edge(1, 2)
    assert comm.broadcast(robots[0], robots, {'test': True}) == 1
    assert robots[1].inbox and not robots[2].inbox
    robots[2].position = np.array([2., 0])
    robots[0].position = np.array([1.5, 0])
    comm.build_graph(robots, 1)
    assert comm.send_message(robots[0], robots[2], {'reconnected': True})


def test_confirmation_counts_same_step_measurements_not_robots_or_timesteps():
    est = Estimator(EstimatorConfig(confirm=3))
    b = est.update(0, 0, Belief(), np.zeros(2), .3, 0)
    for i in range(3):
        b = est.update(0, 0, b, np.array([5., 5.]), .3, 1)
        assert est.relocation_detected(0, 0) == (i == 2)
    np.testing.assert_allclose(b.mean, [5, 5])


def test_conflicting_measurements_can_reset_twice_in_one_batch():
    from sim.messages import observation_message
    strategy = CentralizedStrategy()
    robots = [Robot(i, np.zeros(2)) for i in range(2)]
    strategy.reset(robots, np.random.default_rng(0))
    comm = CommunicationLayer()
    strategy.coordinator_step([observation_message(0, 0, np.zeros(2), .3, 0)], comm, robots, 0)
    strategy.coordinator_step([observation_message(0, 0, np.array([5., 5.]), .3, 1),
                               observation_message(1, 0, np.zeros(2), .3, 1)], comm, robots, 1)
    assert strategy._hub[0].reset_count == 2
    np.testing.assert_allclose(strategy._hub[0].mean, [0, 0])


def test_dispatch_selector_still_uses_truth_not_uncertainty():
    # Characterize the unresolved research placeholder; do not disguise it as
    # an implemented uncertainty-aware algorithm.
    from sim.dispatch import select_robot_for_dispatch
    robots = [Robot(0, np.array([8., 8.])), Robot(1, np.array([2., 2.]))]
    target = Target(0, np.array([9., 9.]))
    robots[0].beliefs[0] = Belief(np.array([2., 2.]), np.eye(2) * 100)
    robots[1].beliefs[0] = Belief(np.array([9., 9.]), np.eye(2) * .001)
    beliefs = {r.id: r.beliefs for r in robots}
    assert select_robot_for_dispatch(robots, beliefs, target, Environment.from_size(12, 12)) == 0
