import numpy as np
import pytest

from sim.estimator import Estimator, EstimatorConfig, chi2_threshold
from sim.robot import Belief

SIGMA = 0.3


def converged_belief(est, target, rng, n=50):
    b = Belief()
    for k in range(n):
        b = est.update(0, 0, b, target + rng.normal(0, SIGMA, 2), SIGMA, 0.1 * k)
    return b


def test_chi2_threshold_two_dof():
    assert chi2_threshold(0.99) == pytest.approx(9.2103, abs=1e-3)
    assert chi2_threshold(0.95) == pytest.approx(5.9915, abs=1e-3)


def test_first_observation_initialises_without_a_reset():
    est = Estimator()
    b = est.update(0, 0, Belief(), np.array([1.0, 2.0]), SIGMA, 0.0)
    np.testing.assert_allclose(b.mean, [1.0, 2.0])
    np.testing.assert_allclose(b.cov, SIGMA ** 2 * np.eye(2))
    assert b.last_seen == 0.0 and b.reset_count == 0 and not est.relocation_detected(0, 0)


def test_noise_is_averaged_away():
    rng = np.random.default_rng(0)
    est = Estimator(EstimatorConfig(process_noise=0.0))
    b = converged_belief(est, np.array([5.0, 5.0]), rng, n=100)
    assert np.linalg.norm(b.mean - [5.0, 5.0]) < 0.1
    assert b.cov[0, 0] < 2 * SIGMA ** 2 / 100  # roughly sigma^2 / n


def test_prediction_inflates_the_covariance():
    est = Estimator(EstimatorConfig(process_noise=0.02))
    b = Belief(mean=np.zeros(2), cov=0.1 * np.eye(2), timestamp=0.0)
    b2 = est.predict(b, 10.0)
    np.testing.assert_allclose(b2.cov, (0.1 + 0.2) * np.eye(2))
    assert b2.timestamp == 10.0


def test_relocation_triggers_a_reset():
    rng = np.random.default_rng(1)
    est = Estimator()
    b = converged_belief(est, np.array([5.0, 5.0]), rng)
    resets_before = b.reset_count  # can be > 0: noise sometimes passes the gate (false resets)
    b = est.update(0, 0, b, np.array([8.0, 5.0]), SIGMA, 10.0)
    assert est.relocation_detected(0, 0)
    np.testing.assert_allclose(b.mean, [8.0, 5.0])
    assert b.reset_count == resets_before + 1 and b.source == "reset" and b.last_seen == 10.0
    assert est.decisions[-1].reset and est.decisions[-1].d2 > est.config.gate


def test_confirm_needs_consecutive_outliers():
    rng = np.random.default_rng(2)
    est = Estimator(EstimatorConfig(confirm=2))
    b = converged_belief(est, np.array([5.0, 5.0]), rng)
    before, resets_before = b.mean.copy(), b.reset_count
    b = est.update(0, 0, b, np.array([8.0, 5.0]), SIGMA, 10.0)  # first outlier: ignored, not fused
    assert not est.relocation_detected(0, 0)
    np.testing.assert_allclose(b.mean, before)
    b = est.update(0, 0, b, np.array([8.1, 5.0]), SIGMA, 10.1)  # second in a row: reset
    assert est.relocation_detected(0, 0) and b.reset_count == resets_before + 1


def test_false_reset_rate_matches_the_gate():
    """With a static target and a converged belief, d2 ~ chi2(2), so about 1 - p of
    the observations are taken for relocations: the false-reset side of the trade-off."""
    rng = np.random.default_rng(3)
    est = Estimator(EstimatorConfig(gate=chi2_threshold(0.95), process_noise=0.0))
    b = converged_belief(est, np.zeros(2), rng, n=200)
    est.reset_log()
    for k in range(4000):
        b = est.update(0, 0, b, rng.normal(0, SIGMA, 2), SIGMA, 20 + 0.1 * k)
    rate = np.mean([d.reset for d in est.decisions])
    assert 0.03 < rate < 0.08  # about 5%, a bit above because a reset restarts from one observation


def test_no_observation_only_predicts():
    est = Estimator(EstimatorConfig(process_noise=0.01))
    b = est.update(0, 0, Belief(), np.zeros(2), SIGMA, 0.0)
    b2 = est.update(0, 0, b, None, SIGMA, 1.0)
    np.testing.assert_allclose(b2.mean, b.mean)
    assert b2.cov[0, 0] > b.cov[0, 0] and not est.relocation_detected(0, 0)
