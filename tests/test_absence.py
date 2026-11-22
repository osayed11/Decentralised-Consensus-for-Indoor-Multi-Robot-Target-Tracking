"""No-sighting rule (decision 3): the expected spot is in view and empty, so the
robot forgets the position but remembers the check."""
import numpy as np
import pytest

from sim import SimConfig, Simulation, messages
from sim.environment import Environment
from sim.estimator import Estimator, EstimatorConfig
from sim.robot import Belief, Robot
from sim.strategies import DecentralizedConsensusStrategy, IndependentStrategy

HERE = np.array([1.0, 0.0])      # where the robot stands for the check


def confident(mean=(0.0, 0.0), last_seen=5.0):
    return Belief(mean=np.array(mean, float), cov=0.09 * np.eye(2), timestamp=last_seen, last_seen=last_seen)


def check_empty(est, belief, t=10.0):
    return est.update(0, 0, belief, None, 0.3, t, in_view=True, view_pos=HERE, view_radius=2.4)


def test_in_view_and_empty_clears_the_belief_but_keeps_the_check():
    est = Estimator(EstimatorConfig())
    b = check_empty(est, confident())
    assert not b.valid and b.cov is None
    assert b.last_absence == 10.0 and b.last_seen == 5.0 and b.source == "absence"
    np.testing.assert_allclose(b.absence_pos, HERE)
    assert b.absence_radius == 2.4
    assert est.absence_detected(0, 0) and est.absences == [(10.0, 0, 0)]


def test_fires_once_then_nothing_to_check():
    est = Estimator(EstimatorConfig())
    b = check_empty(est, confident())
    b2 = check_empty(est, b, t=10.1)
    assert not est.absence_detected(0, 0) and len(est.absences) == 1
    assert b2.last_absence == 10.0


def test_not_in_view_or_switched_off_or_seen_leaves_the_belief():
    est = Estimator(EstimatorConfig())
    b = est.update(0, 0, confident(), None, 0.3, 10.0, in_view=False)
    assert b.valid and not est.absence_detected(0, 0)
    off = Estimator(EstimatorConfig(absence=False))
    assert check_empty(off, confident()).valid
    b = est.update(0, 0, confident(), np.array([0.1, 0.0]), 0.3, 10.0, in_view=True)   # seen -> gating path
    assert b.valid and b.source == "own_obs" and not est.absence_detected(0, 0)


def test_next_sighting_starts_fresh_without_a_reset_and_keeps_the_check():
    est = Estimator(EstimatorConfig())
    b = check_empty(est, confident())
    b = est.update(0, 0, b, np.array([15.0, 0.0]), 0.3, 12.0)
    assert b.valid and not est.relocation_detected(0, 0)
    np.testing.assert_allclose(b.mean, [15.0, 0.0])
    assert b.last_seen == 12.0 and b.last_absence == 10.0


def test_without_the_rule_the_same_sighting_counts_as_a_reset():
    # static model: with the resting/carried model a 15 m move in 7 unseen
    # seconds is explainable as a carry and is tracked without a reset
    est = Estimator(EstimatorConfig(absence=False, model="static"))
    b = check_empty(est, confident())
    est.update(0, 0, b, np.array([15.0, 0.0]), 0.3, 12.0)
    assert est.relocation_detected(0, 0)


# ---- the in-view check --------------------------------------------------------
def strategy_with(env=None, los=False, cls=IndependentStrategy, **kw):
    s = cls(**kw)
    s.reset([], np.random.default_rng(0))
    s.bind(env, los)
    return s


def test_in_view_uses_a_share_of_sensor_range():
    s = strategy_with(absence_view_fraction=0.8)
    r = Robot(id=0, position=np.zeros(2), sensor_range=3.0)
    assert s.belief_in_view(r, confident((2.3, 0.0)))
    assert not s.belief_in_view(r, confident((2.5, 0.0)))   # 2.5 > 0.8 * 3
    assert not s.belief_in_view(r, Belief())


def test_a_wall_is_not_an_absence_when_line_of_sight_is_on():
    grid = np.zeros((20, 20), dtype=int)
    grid[:, 10] = 1                                         # wall at x = 5 m
    env = Environment.from_array(grid, resolution=0.5, robot_radius=0.0)
    r = Robot(id=0, position=np.array([4.0, 5.0]), sensor_range=3.0)
    behind_wall = confident((6.0, 5.0))
    assert strategy_with(env, los=False).belief_in_view(r, behind_wall)
    assert not strategy_with(env, los=True).belief_in_view(r, behind_wall)


def test_strategy_records_the_robots_view_area():
    s = strategy_with(absence_view_fraction=0.8)
    r = Robot(id=0, position=np.array([2.0, 2.0]), sensor_range=3.0)
    r.beliefs[0] = confident((2.5, 2.0))
    b = s.update_belief(r, 0, None, [], 10.0)
    assert not b.valid
    np.testing.assert_allclose(b.absence_pos, [2.0, 2.0])
    assert b.absence_radius == pytest.approx(2.4)


# ---- messages and contradiction ---------------------------------------------
def cleared():
    return Estimator().apply_absence(confident(), 10.0, HERE, 2.4)


def test_absence_message_carries_the_checked_area():
    a = messages.absence_message(1, 0, cleared(), 10.5)
    assert a["type"] == messages.ABSENCE and a["t_obs"] == 10.0 and a["radius"] == 2.4
    np.testing.assert_allclose(a["pos"], HERE)
    assert messages.for_target([a], 0, messages.ABSENCE) == [a]


def test_contradiction_only_for_older_claims_inside_the_checked_area():
    b = cleared()                                                               # area: 2.4 m around (1, 0) at t=10
    assert messages.belief_contradicted(np.array([0.0, 0.0]), 5.0, b)           # stale kitchen rumour
    assert not messages.belief_contradicted(np.array([8.0, 0.0]), 5.0, b)       # elsewhere: fine
    assert not messages.belief_contradicted(np.array([0.0, 0.0]), 11.0, b)      # newer sighting: fine
    assert not messages.belief_contradicted(np.array([0.0, 0.0]), 5.0, confident())  # no check made


def test_decentralised_robot_shares_its_check_when_it_has_no_belief():
    s = strategy_with(cls=DecentralizedConsensusStrategy)
    r = Robot(id=3, position=HERE.copy())
    r.beliefs[0] = cleared()
    r.beliefs[1] = confident((5.0, 5.0))
    sent = s.get_messages_to_send(r, None, [r], 10.5)
    kinds = {(p["target_id"], p["type"]) for _, p in sent}
    assert kinds == {(0, messages.ABSENCE), (1, messages.BELIEF)}


# ---- full simulation ------------------------------------------------------------
def test_stale_belief_is_cleared_in_a_running_simulation():
    """Plant a confident belief right next to robot 0, far from the real target:
    the first step must clear it and the trial must log one event."""
    cfg = SimConfig(time_limit=1.0, seed=0, query_interval=None, n_robots=1, relocation_dist="none")
    sim = Simulation(cfg)
    r, tg = sim.robots[0], sim.targets[0]
    if np.linalg.norm(tg.position - r.position) <= r.sensor_range + 1.0:
        tg.position = sim.env.random_free_position(np.random.default_rng(1), [r.position], r.sensor_range + 3)
    r.beliefs[0] = confident(tuple(r.position + [0.5, 0.0]), last_seen=0.0)
    sim.run()
    assert (sim.events_dataframe()["event"] == "absence_detected").sum() == 1
    assert sim.summary()["n_absence_detections"] == 1
    assert not r.beliefs[0].valid and r.beliefs[0].last_absence == 0.0
    df = sim.step_dataframe()
    centre = sim.env.centre
    last = df.iloc[-1]
    assert last["error_eval"] == pytest.approx(np.hypot(last.true_x - centre[0], last.true_y - centre[1]))
