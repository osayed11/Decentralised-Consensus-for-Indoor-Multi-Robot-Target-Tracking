"""Unexpected failures: robots and the central server (decided 2026-11-22)."""
import numpy as np
import pandas as pd
import pytest

from sim import SimConfig, Simulation
from sim.comms import COORDINATOR, CommunicationLayer
from sim.faults import FaultSchedule, sample_failure_time
from sim.metrics import failure_impact
from sim.robot import Belief, Robot
from sim.strategies import IndependentStrategy


def make_robot(i, x):
    return Robot(id=i, position=np.array([x, 0.0]), comm_range=1.0)


def run(time_limit=120.0, seed=3, **kw):
    cfg = SimConfig(time_limit=time_limit, seed=seed, query_interval=None, **kw)
    sim = Simulation(cfg)
    sim.run()
    return sim


def event_times(sim, name):
    ev = sim.events_dataframe()
    return ev[ev["event"] == name] if not ev.empty else ev


# ---- the failure-time draw -------------------------------------------------
def test_constant_risk_matches_the_requested_probability():
    rng = np.random.default_rng(0)
    T, p = 300.0, 0.3
    times = np.array([sample_failure_time(rng.random(), p, T) for _ in range(40_000)])
    failed = times[np.isfinite(times)]
    assert len(failed) / len(times) == pytest.approx(p, abs=0.01)
    assert failed.min() >= 0 and failed.max() < T
    # constant hazard: P(fail in first half | fail) = (1-(1-p)^0.5)/p, not 0.5
    expected_first_half = (1 - (1 - p) ** 0.5) / p
    assert (failed < T / 2).mean() == pytest.approx(expected_first_half, abs=0.01)


def test_zero_probability_never_fails_and_one_is_rejected():
    assert sample_failure_time(0.0, 0.0, 300.0) == float("inf")
    with pytest.raises(ValueError):
        sample_failure_time(0.5, 1.0, 300.0)


def test_fixed_time_overrides_the_random_draw():
    fs = FaultSchedule(robot_failures={1: 42.0}, robot_failure_prob=0.0, horizon=300.0)
    fs.sample([0, 1, 2], np.random.default_rng(0))
    assert fs.robot_schedule == {1: (42.0, float("inf"))}


# ---- fairness ---------------------------------------------------------------
def test_every_strategy_gets_the_same_failures():
    sched = []
    for strategy in ("independent", "centralized", "decentralized"):
        sim = Simulation(SimConfig(strategy=strategy, seed=11, robot_failure_prob=0.6, coordinator_failure_prob=0.6))
        sched.append((sim.faults.robot_schedule, sim.faults.coordinator_schedule))
    assert sched[0] == sched[1] == sched[2]
    assert sched[0][0], "expected at least one robot failure at p=0.6"


def test_failures_do_not_change_the_rest_of_the_world():
    """With local collision interactions disabled, faults do not affect the
    other robots' patrol trajectories or the target relocation RNG.
    With safety enabled, a failed robot can physically obstruct another robot,
    so unchanged trajectories are neither expected nor desirable.
    """
    base = run(relocation_min=5.0, relocation_max=15.0, robot_avoidance=False)
    faulty = run(relocation_min=5.0, relocation_max=15.0, robot_failures={1: 20.0}, robot_avoidance=False)
    r_base = event_times(base, "relocation")[["t", "x", "y"]].to_numpy()
    r_faulty = event_times(faulty, "relocation")[["t", "x", "y"]].to_numpy()
    assert len(r_base) >= 3
    np.testing.assert_allclose(r_base, r_faulty)
    a, b = base.step_dataframe(), faulty.step_dataframe()
    for rid in (0, 2):
        np.testing.assert_allclose(a[a.robot_id == rid][["robot_x", "robot_y"]].to_numpy(),
                                   b[b.robot_id == rid][["robot_x", "robot_y"]].to_numpy())


# ---- robot failure and recovery --------------------------------------------
def test_permanent_failure_is_logged_once_and_robot_stays_down():
    sim = run(robot_failures={1: 20.0})
    fails = event_times(sim, "robot_failure")
    assert list(fails["robot_id"]) == [1] and fails["t"].iloc[0] == pytest.approx(20.0)
    assert sim.robots[1].failed and event_times(sim, "robot_recovery").empty
    assert sim.summary()["n_robot_failures"] == 1


@pytest.mark.parametrize("keep", [True, False])
def test_temporary_outage_comes_back_with_or_without_memory(keep):
    robot = make_robot(0, 0.0)
    stale = Belief(mean=np.zeros(2), cov=np.eye(2), timestamp=0.0, last_seen=0.0)
    robot.beliefs[0] = stale
    fs = FaultSchedule(robot_failures={0: 1.0}, robot_downtime=5.0, keep_belief_on_recovery=keep)
    fs.sample([0], np.random.default_rng(0))
    assert [e["event"] for e in fs.apply(1.0, [robot])] == ["robot_failure"] and robot.failed
    assert fs.apply(5.9, [robot]) == [] and robot.failed            # still down
    assert [e["event"] for e in fs.apply(6.0, [robot])] == ["robot_recovery"]
    assert not robot.failed and robot.mode == "patrol"
    assert (robot.beliefs.get(0) is stale) == keep                  # stale belief kept, or wiped by a reboot


def test_temporary_outage_in_a_full_run():
    sim = run(time_limit=40.0, robot_failures={0: 10.0}, robot_downtime=5.0)
    assert not sim.robots[0].failed
    ev = sim.events_dataframe()
    assert list(ev[ev["event"].str.startswith("robot_")]["event"]) == ["robot_failure", "robot_recovery"]
    df = sim.step_dataframe()
    down = df[(df.robot_id == 0) & (df.t >= 10.0) & (df.t < 15.0)]
    assert down["robot_failed"].all() and down["error_eval"].isna().all()   # excluded while down


# ---- coordinator link and server failure -----------------------------------
def test_coordinator_is_reachable_from_any_distance_and_replies_to_all():
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    a, b = make_robot(0, 0.0), make_robot(1, 500.0)        # far apart, no robot-robot link
    comm.build_graph([a, b], 0.0)
    assert comm.send_to_coordinator(b, {"x": 1.0})
    assert len(comm.coordinator_inbox) == 1
    assert comm.coordinator_broadcast([a, b], {"y": 2.0}) == 2
    assert a.inbox == [{"y": 2.0}] and b.inbox == [{"y": 2.0}]
    assert {m.receiver for m in comm.log} >= {COORDINATOR} and any(m.sender == COORDINATOR for m in comm.log)


def test_server_down_receives_and_sends_nothing_but_attempts_are_counted():
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    a = make_robot(0, 0.0)
    comm.set_coordinator_up(False)
    assert not comm.send_to_coordinator(a, {"x": 1.0})
    assert comm.coordinator_inbox == [] and comm.coordinator_broadcast([a], {"y": 1}) == 0
    assert comm.messages_sent() == 1 and comm.log[0].reason == "coordinator_down"


def test_packet_loss_applies_to_the_coordinator_link():
    comm = CommunicationLayer(packet_loss_prob=0.5, rng=np.random.default_rng(1))
    a = make_robot(0, 0.0)
    ok = sum(comm.send_to_coordinator(a, {"x": 1.0}) for _ in range(4000))
    assert ok / 4000 == pytest.approx(0.5, abs=0.03)


class _EchoCentral(IndependentStrategy):
    """Minimal stand-in for a centralised strategy: robots ping the server and
    the server answers each step.  Only exercises the plumbing."""
    name = "echo"

    def get_messages_to_send(self, robot, comm, robots, t):
        return [(COORDINATOR, {"type": "ping", "sender": robot.id, "t": t})]

    def coordinator_step(self, inbox, comm, robots, t):
        return [{"type": "pong", "n": len(inbox), "t": t}] if inbox else []


def test_server_failure_cuts_the_round_trip_and_recovery_restores_it():
    cfg = SimConfig(time_limit=30.0, seed=0, query_interval=None,
                    coordinator_failure_time=10.0, coordinator_downtime=10.0)
    sim = Simulation(cfg, strategy=_EchoCentral())
    sim.strategy.reset(sim.robots, sim.rng)
    sim.run()
    down = [m for m in sim.comm.log if 10.0 <= m.t < 20.0]
    up_late = [m for m in sim.comm.log if m.t >= 20.0]
    assert down and not any(m.delivered for m in down)
    assert not any(m.sender == COORDINATOR for m in down)
    assert any(m.delivered and m.sender == COORDINATOR for m in up_late)
    names = list(sim.events_dataframe()["event"])
    assert "coordinator_failure" in names and "coordinator_recovery" in names


# ---- failure-impact metric ---------------------------------------------------
def test_failure_impact_uses_survivors_and_finds_recovery():
    rows = []
    for t in np.arange(0, 60, 1.0):
        # survivors (robots 0, 2): 1 m before t=20, 3 m for 10 s after, then back to 1 m
        e = 3.0 if 20 <= t < 30 else 1.0
        rows += [{"t": t, "target_id": 0, "robot_id": 0, "error_eval": e},
                 {"t": t, "target_id": 0, "robot_id": 2, "error_eval": e},
                 {"t": t, "target_id": 0, "robot_id": 1, "error_eval": 50.0 if t < 20 else np.nan}]
    df = pd.DataFrame(rows)
    ev = pd.DataFrame([{"t": 20.0, "event": "robot_failure", "robot_id": 1}])
    (r,) = failure_impact(df, ev, before=10.0, after=30.0)
    assert r["err_before"] == pytest.approx(1.0)          # robot 1's 50 m is not in the baseline
    assert r["err_after"] == pytest.approx((10 * 3 + 21 * 1) / 31)
    assert r["recovery_time"] == pytest.approx(10.0)


def test_summary_reports_failure_impact():
    s = run(robot_failures={1: 30.0}, coordinator_failure_time=60.0).summary()
    assert s["n_robot_failures"] == 1 and s["n_coordinator_failures"] == 1
    for k in ("robot_failure_err_before", "robot_failure_increase", "robot_failure_recovery_time",
              "coordinator_failure_err_after"):
        assert k in s
    assert not np.isnan(s["robot_failure_err_before"])
