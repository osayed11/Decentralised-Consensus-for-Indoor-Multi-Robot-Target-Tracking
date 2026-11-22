"""Accuracy and search-efficiency metrics computed from the per-step log.

Definitions (decision 1, see README "Metrics"):

* Every alive robot holds its own guess of each target.  A robot's error is the
  distance from its belief mean to the true target position.
* A robot with no belief yet is scored as if it guessed the centre of the
  building (``Environment.centre``): it is not dropped, so a lone informed
  robot cannot stand in for the whole team.
* A failed robot is excluded from the moment it fails.
* Per time step the alive robots' errors are combined in four ways:
    mean    average error           -> headline (``rmse_mean``)
    frac    share within tolerance  -> companion (``frac_robots_localised``)
    worst   largest error           -> robustness view (``rmse_worst``)
    best    smallest error          -> "does anyone know" (``rmse_best``)
  RMSE = sqrt(time-average of the per-step mean of squared errors).
* Search efficiency: first time *every* alive robot is within tolerance
  (``time_all_localised``, headline) and first time *any* robot is
  (``time_any_localised``), each with a success flag before the time limit.
  With several targets, the time is when the last target reached that state.

Failure impact (unexpected failures, see ``failure_impact``): for each robot or
coordinator failure, the team's mean error in a window just before vs just
after it, and how long until it is back to the pre-failure level.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def eval_error_column(df: pd.DataFrame) -> pd.Series:
    """Error used for metrics: ``error_eval`` if logged, else raw ``error``
    (older logs) with failed robots masked out."""
    if "error_eval" in df:
        return df["error_eval"]
    err = df["error"].copy()
    if "robot_failed" in df:
        err[df["robot_failed"].astype(bool)] = np.nan
    return err


def per_step(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """One row per (t, target_id) with mean / mse / worst / best / frac / n_alive."""
    d = df.assign(_e=eval_error_column(df)).dropna(subset=["_e"])
    if d.empty:
        return pd.DataFrame(columns=["t", "target_id", "mean", "mse", "worst", "best", "frac", "n_alive"])
    d = d.assign(_sq=d["_e"] ** 2, _in=(d["_e"] <= threshold).astype(float))
    g = d.groupby(["t", "target_id"])
    out = pd.DataFrame({
        "mean": g["_e"].mean(), "mse": g["_sq"].mean(), "worst": g["_e"].max(),
        "best": g["_e"].min(), "frac": g["_in"].mean(), "n_alive": g["_e"].size(),
    })
    return out.reset_index()


def _first_time(steps: pd.DataFrame, mask: pd.Series) -> float:
    """Per target, first t where mask holds; the max over targets (NaN if any never)."""
    times = []
    for tid in steps["target_id"].unique():
        sel = steps[(steps["target_id"] == tid) & mask]
        if sel.empty:
            return np.nan
        times.append(float(sel["t"].min()))
    return max(times) if times else np.nan


def accuracy_metrics(df: pd.DataFrame, threshold: float = 1.0) -> Dict[str, float]:
    steps = per_step(df, threshold)
    if steps.empty:
        return {}
    has_belief = df.loc[eval_error_column(df).notna(), "has_belief"] if "has_belief" in df else None
    t_all = _first_time(steps, steps["worst"] <= threshold)
    t_any = _first_time(steps, steps["best"] <= threshold)
    last_t = steps["t"].max()
    final = steps[steps["t"] == last_t]
    return {
        "rmse_mean": float(np.sqrt(steps["mse"].mean())),
        "mean_error": float(steps["mean"].mean()),
        "rmse_worst": float(np.sqrt((steps["worst"] ** 2).mean())),
        "rmse_best": float(np.sqrt((steps["best"] ** 2).mean())),
        "frac_robots_localised": float(steps["frac"].mean()),
        "frac_robots_with_belief": float(has_belief.mean()) if has_belief is not None and len(has_belief) else np.nan,
        "time_all_localised": t_all,
        "search_success_all": float(not np.isnan(t_all)),
        "time_any_localised": t_any,
        "search_success_any": float(not np.isnan(t_any)),
        "final_error_mean": float(final["mean"].mean()),
    }


def _team_mean_series(df: pd.DataFrame, exclude_robot: int = None) -> pd.Series:
    """Per-step team mean error (averaged over targets), optionally without one robot."""
    d = df if exclude_robot is None else df[df["robot_id"] != exclude_robot]
    steps = per_step(d)
    if steps.empty:
        return pd.Series(dtype=float)
    return steps.groupby("t")["mean"].mean()


def failure_impact(df: pd.DataFrame, events: pd.DataFrame, before: float = 10.0,
                   after: float = 30.0) -> List[Dict[str, float]]:
    """One row per failure event.

    * robot failure: measured on the *surviving* robots (the failed one is left
      out of both windows), so the number is the effect on the rest of the team;
    * coordinator failure: measured on all alive robots.

    ``err_before`` = mean team error in [t_f - before, t_f), ``err_after`` = in
    [t_f, t_f + after], ``increase`` = after - before, ``recovery_time`` = time
    from t_f until the team error is first back at or below ``err_before``
    (NaN if never within the trial).  Windows are clipped at the trial edges.
    """
    if events is None or events.empty or df.empty:
        return []
    rows = []
    fails = events[events["event"].isin(["robot_failure", "coordinator_failure"])]
    for _, e in fails.iterrows():
        t_f = float(e["t"])
        robot = int(e["robot_id"]) if e["event"] == "robot_failure" else None
        series = _team_mean_series(df, exclude_robot=robot)
        pre = series[(series.index >= t_f - before) & (series.index < t_f)]
        post = series[(series.index >= t_f) & (series.index <= t_f + after)]
        err_before = float(pre.mean()) if len(pre) else np.nan
        err_after = float(post.mean()) if len(post) else np.nan
        rec = np.nan
        if not np.isnan(err_before):
            back = series[(series.index >= t_f) & (series <= err_before + 1e-9)]
            rec = float(back.index[0] - t_f) if len(back) else np.nan
        rows.append({"t": t_f, "kind": "robot" if robot is not None else "coordinator", "robot_id": robot,
                     "err_before": err_before, "err_after": err_after,
                     "increase": err_after - err_before, "recovery_time": rec})
    return rows


def failure_impact_summary(df: pd.DataFrame, events: pd.DataFrame, before: float = 10.0,
                           after: float = 30.0) -> Dict[str, float]:
    """Per-trial means of ``failure_impact`` rows, separately for robot and
    coordinator failures (NaN when the trial had none of that kind)."""
    rows = pd.DataFrame(failure_impact(df, events, before, after))
    out: Dict[str, float] = {}
    for kind in ("robot", "coordinator"):
        sel = rows[rows["kind"] == kind] if not rows.empty else rows
        for col in ("err_before", "err_after", "increase", "recovery_time"):
            out[f"{kind}_failure_{col}"] = float(sel[col].mean()) if len(sel) else np.nan
        out[f"{kind}_failure_recovered"] = float(sel["recovery_time"].notna().mean()) if len(sel) else np.nan
    return out
