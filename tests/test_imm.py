"""Resting / carried (IMM) estimator."""
import numpy as np
import pytest

from sim.estimator import Estimator, EstimatorConfig
from sim.harness import SimConfig, Simulation
from sim.robot import Belief

SIGMA = 0.3
DT = 0.1


def track(est, positions, rng, t0=0.0, b=None):
    b = Belief() if b is None else b
    for k, p in enumerate(positions):
        b = est.update(0, 0, b, np.asarray(p) + rng.normal(0, SIGMA, 2), SIGMA, t0 + DT * k)
    return b


def walk(start, velocity, n):
    return [np.asarray(start) + np.asarray(velocity) * DT * k for k in range(n)]


def test_resting_target_stays_in_resting_mode():
    rng = np.random.default_rng(0)
    est = Estimator()
    b = track(est, [np.array([5.0, 5.0])] * 100, rng)
    assert b.p_moving < 0.1
    assert np.linalg.norm(b.mean - [5.0, 5.0]) < 0.15
    assert np.linalg.norm(b.velocity) < 0.2


def walk_trial(model, seed):
    rng = np.random.default_rng(seed)
    est = Estimator(EstimatorConfig(model=model))
    b = track(est, [np.array([2.0, 7.0])] * 50, rng)                  # resting
    return track(est, walk([2.0, 7.0], [1.2, 0.0], 80), rng, t0=5.0, b=b)  # carried along the corridor


def test_carried_target_is_followed():
    b = walk_trial("imm", 0)
    assert b.p_moving > 0.9
    assert b.velocity[0] == pytest.approx(1.2, abs=0.3)
    assert np.linalg.norm(b.mean - [2.0 + 1.2 * DT * 79, 7.0]) < 0.5


def test_carried_target_causes_far_fewer_resets_than_the_static_model():
    imm = np.mean([walk_trial("imm", s).reset_count for s in range(40)])
    static = np.mean([walk_trial("static", s).reset_count for s in range(40)])
    assert imm < 1.5 and static > 5 * imm


def test_put_down_returns_to_resting_mode():
    rng = np.random.default_rng(2)
    est = Estimator()
    b = track(est, walk([2.0, 7.0], [1.2, 0.0], 60), rng)
    stop = np.array([2.0 + 1.2 * DT * 59, 7.0])
    b = track(est, [stop] * 100, rng, t0=6.0, b=b)
    assert b.p_moving < 0.2
    assert np.linalg.norm(b.mean - stop) < 0.2


def test_prediction_moves_with_velocity_and_uncertainty_stays_bounded():
    rng = np.random.default_rng(3)
    est = Estimator()
    b = track(est, walk([2.0, 7.0], [1.2, 0.0], 60), rng)
    b2 = est.predict(b, b.timestamp + 1.0)
    assert b2.mean[0] > b.mean[0] + 0.5          # coasts forward while unseen
    far = est.predict(b, b.timestamp + 600.0)   # ten minutes unseen
    assert np.sqrt(np.max(np.diag(far.cov))) <= est.config.max_pos_std + 1e-6
    assert 0.0 < far.p_moving < 0.5             # tends to the long-run share of time carried


def test_unexplained_jump_still_resets():
    rng = np.random.default_rng(4)
    est = Estimator()
    b = track(est, [np.array([2.0, 2.0])] * 50, rng)
    b = est.update(0, 0, b, np.array([18.0, 12.0]), SIGMA, b.timestamp + DT)  # 20 m in 0.1 s
    assert est.relocation_detected(0, 0) and b.reset_count == 1
    np.testing.assert_allclose(b.mean, [18.0, 12.0])


def test_outside_change_to_the_belief_recentres_both_modes():
    """A consensus step may overwrite mean/cov; the next update must start from it."""
    rng = np.random.default_rng(5)
    est = Estimator()
    b = track(est, [np.array([5.0, 5.0])] * 50, rng)
    from dataclasses import replace
    moved = replace(b, mean=np.array([9.0, 5.0]), cov=0.05 * np.eye(2))
    b2 = est.predict(moved, moved.timestamp + DT)
    assert np.linalg.norm(b2.mean - [9.0, 5.0]) < 0.1


def test_belief_without_imm_state_is_accepted():
    est = Estimator()
    b = Belief(mean=np.zeros(2), cov=0.1 * np.eye(2), timestamp=0.0, last_seen=0.0)
    b = est.update(0, 0, b, np.array([0.1, 0.0]), SIGMA, 0.1)
    assert b.valid and 0.0 < b.p_moving < 1.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        EstimatorConfig(model="kalman")


def test_sim_runs_with_both_models_and_logs_p_moving():
    for model in ("imm", "static"):
        cfg = SimConfig(time_limit=60.0, strategy_kwargs={"model": model})
        sim = Simulation(cfg, seed=0)
        sim.run()
        df = sim.step_dataframe()
        if model == "imm":
            assert df.loc[df.has_belief, "p_moving"].between(0, 1).all()
        else:
            assert df["p_moving"].isna().all()
