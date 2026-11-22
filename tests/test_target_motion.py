"""Carried-target motion: walks a free path at carry speed, then rests."""
import numpy as np
import pytest

from sim.environment import Environment
from sim.harness import SimConfig, Simulation
from sim.target import Target

ENV = Environment()


def make_target(**kw):
    rng = np.random.default_rng(0)
    tg = Target(id=0, position=ENV.random_free_position(rng), interval_dist="fixed", interval_mean=5.0, **kw)
    tg.schedule_next(0.0, rng)
    return tg, rng


def run(tg, rng, until, dt=0.1):
    out = []
    for k in range(int(round(until / dt)) + 1):
        t = k * dt
        tg.step(t, ENV, rng)
        out.append((t, tg.position.copy(), tg.moving))
    return out


def test_carried_target_walks_at_carry_speed_through_free_space():
    tg, rng = make_target(carry_speed=1.0)
    trace = run(tg, rng, until=60.0)
    moving = [(t, p) for t, p, m in trace if m]
    assert moving, "target never moved"
    for (_, a, _), (_, b, _) in zip(trace, trace[1:]):
        assert np.linalg.norm(b - a) <= 1.0 * 0.1 + 1e-9
    assert all(ENV.is_free(p) for _, p, _ in trace)


def test_carried_target_arrives_at_destination_then_rests():
    tg, rng = make_target(carry_speed=1.2)
    run(tg, rng, until=120.0)
    assert len(tg.history) >= 2
    ev = tg.history[0]
    assert ev.t_arrive > ev.t
    walked = sum(np.linalg.norm(b - a) for a, b in zip([ev.old_pos] + ev.path, ev.path))
    assert ev.t_arrive - ev.t == pytest.approx(walked / 1.2, abs=0.11)
    # the next move starts a fixed rest after arrival
    assert tg.history[1].t == pytest.approx(ev.t_arrive + 5.0, abs=0.11)


def test_teleport_mode_jumps_instantly():
    tg, rng = make_target(motion="teleport")
    trace = run(tg, rng, until=6.0)
    assert not any(m for _, _, m in trace)
    ev = tg.history[0]
    assert ev.t_arrive == ev.t and np.allclose(tg.position, ev.new_pos)


def test_pause_delays_departure_but_not_an_ongoing_walk():
    tg, rng = make_target()
    tg.pause_relocation()
    run(tg, rng, until=8.0)
    assert not tg.history  # held while paused
    tg.resume_relocation()
    tg.step(8.1, ENV, rng)
    assert tg.moving
    tg.pause_relocation()
    p = tg.position.copy()
    tg.step(8.2, ENV, rng)
    assert not np.allclose(p, tg.position)  # the carrier keeps walking


def test_bad_motion_rejected():
    with pytest.raises(ValueError):
        Target(id=0, position=np.zeros(2), motion="fly")


def test_sim_logs_transit_and_robots_can_see_target_in_transit():
    cfg = SimConfig(n_robots=6, sensor_range=4.0, time_limit=300.0, relocation_min=20.0, relocation_max=40.0)
    seen_moving = False
    for seed in range(5):
        sim = Simulation(cfg, seed=seed)
        sim.run()
        df, ev = sim.step_dataframe(), sim.events_dataframe()
        assert df["target_moving"].any()
        assert (ev["event"] == "arrival").sum() >= (ev["event"] == "relocation").sum() - 1
        seen_moving |= bool((df["target_moving"] & df["observed"]).any())
    assert seen_moving


def test_carried_motion_is_reproducible():
    cfg = SimConfig(time_limit=120.0)
    a, b = Simulation(cfg, seed=3), Simulation(cfg, seed=3)
    a.run(); b.run()
    assert np.allclose(a.step_dataframe()[["true_x", "true_y"]], b.step_dataframe()[["true_x", "true_y"]])
