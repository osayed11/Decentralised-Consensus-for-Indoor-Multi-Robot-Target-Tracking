"""Accuracy / search-efficiency metric definitions (decision 1)."""
import numpy as np
import pandas as pd
import pytest

from sim.environment import Environment
from sim.harness import SimConfig, Simulation
from sim.metrics import accuracy_metrics, per_step


def _row(t, rid, err, has_belief=True, failed=False):
    return {"t": t, "target_id": 0, "robot_id": rid, "error_eval": np.nan if failed else err,
            "has_belief": has_belief, "robot_failed": failed}


def test_mean_sees_what_best_hides():
    # A knows (0.1 m), B and C are 8 m off: best says 0.1, mean says ~5.4
    df = pd.DataFrame([_row(0.0, 0, 0.1), _row(0.0, 1, 8.0), _row(0.0, 2, 8.0)])
    m = accuracy_metrics(df, threshold=1.0)
    assert m["rmse_best"] == pytest.approx(0.1)
    assert m["mean_error"] == pytest.approx((0.1 + 8 + 8) / 3)
    assert m["rmse_mean"] == pytest.approx(np.sqrt((0.01 + 64 + 64) / 3))
    assert m["frac_robots_localised"] == pytest.approx(1 / 3)
    assert m["rmse_worst"] == pytest.approx(8.0)


def test_failed_robots_are_excluded():
    df = pd.DataFrame([_row(0.0, 0, 0.5), _row(0.0, 1, 99.0, failed=True)])
    m = accuracy_metrics(df)
    assert m["mean_error"] == pytest.approx(0.5)
    assert per_step(df)["n_alive"].iloc[0] == 1


def test_search_time_all_versus_any():
    rows = [_row(0.0, 0, 5.0), _row(0.0, 1, 5.0),
            _row(1.0, 0, 0.5), _row(1.0, 1, 5.0),     # one robot knows
            _row(2.0, 0, 0.5), _row(2.0, 1, 0.8)]     # everyone knows
    m = accuracy_metrics(pd.DataFrame(rows), threshold=1.0)
    assert m["time_any_localised"] == 1.0
    assert m["time_all_localised"] == 2.0
    assert m["search_success_all"] == 1.0


def test_never_localised_is_nan_and_unsuccessful():
    m = accuracy_metrics(pd.DataFrame([_row(0.0, 0, 5.0), _row(1.0, 0, 4.0)]))
    assert np.isnan(m["time_all_localised"]) and m["search_success_all"] == 0.0


def test_robot_without_belief_scores_as_building_centre():
    cfg = SimConfig(time_limit=0.3, n_robots=3, query_interval=None, strategy="independent")
    sim = Simulation(cfg, seed=0)
    sim.run()
    df = sim.step_dataframe()
    no_b = df[~df["has_belief"] & ~df["robot_failed"]]
    assert len(no_b), "expected at least one robot without a belief at the start"
    centre = sim.env.centre
    expected = np.hypot(no_b["true_x"] - centre[0], no_b["true_y"] - centre[1])
    assert np.allclose(no_b["error_eval"], expected)
    assert no_b["error"].isna().all()           # raw error stays NaN


def test_centre_is_in_reachable_space_of_default_plan():
    env = Environment()
    assert env.is_reachable(env.centre)


def test_summary_has_new_metrics_and_old_names_are_gone():
    sim = Simulation(SimConfig(time_limit=5.0, query_interval=None), seed=1)
    sim.run()
    s = sim.summary()
    for k in ("rmse_mean", "frac_robots_localised", "rmse_worst", "rmse_best",
              "time_all_localised", "search_success_all", "time_any_localised"):
        assert k in s
    assert "rmse_team" not in s
