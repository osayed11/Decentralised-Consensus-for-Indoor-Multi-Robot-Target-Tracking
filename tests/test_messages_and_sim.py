import numpy as np

from sim import SimConfig, Simulation
from sim import messages
from sim.robot import Belief


def test_belief_message_round_trip_and_filtering():
    b = Belief(mean=np.array([1.0, 2.0]), cov=np.eye(2), timestamp=3.0, last_seen=2.5, reset_count=4)
    msg = messages.belief_message(7, 0, b, 3.0)
    assert set(msg) == {"type", "sender", "target_id", "t_sent", "mean", "cov", "last_seen", "reset_count"}
    back = messages.message_to_belief(msg)
    np.testing.assert_allclose(back.mean, b.mean)
    assert back.last_seen == 2.5 and back.source == "msg"
    obs = messages.observation_message(7, 1, np.zeros(2), 0.3, 3.0)
    assert messages.for_target([msg, obs], 0) == [msg]
    assert messages.for_target([msg, obs], 1, messages.OBSERVATION) == [obs]


def test_is_newer_compares_sightings():
    mine = Belief(mean=np.zeros(2), cov=np.eye(2), last_seen=10.0)
    assert messages.is_newer({"last_seen": 12.0}, mine)
    assert not messages.is_newer({"last_seen": 8.0}, mine)
    assert messages.is_newer({"last_seen": 0.0}, Belief())


def run(strategy, **overrides):
    cfg = SimConfig(strategy=strategy, time_limit=60.0, seed=5, query_interval=None, **overrides)
    sim = Simulation(cfg)
    sim.run()
    return sim


def test_decentralized_messages_flow_and_independent_stays_silent():
    dec, ind = run("decentralized"), run("independent")
    assert dec.comm.messages_delivered() > 0
    assert ind.comm.messages_sent() == 0


def test_every_strategy_sees_the_same_world():
    """Same seed -> same target relocations, whatever each strategy sends."""
    relocations = []
    for strategy in ("independent", "centralized", "decentralized"):
        events = run(strategy, packet_loss_prob=0.3, relocation_min=5.0, relocation_max=15.0).events_dataframe()
        relocations.append(events[events["event"] == "relocation"][["t", "x", "y"]].to_numpy())
    assert len(relocations[0]) >= 3
    np.testing.assert_allclose(relocations[0], relocations[1])
    np.testing.assert_allclose(relocations[0], relocations[2])


def test_failed_robot_stops_communicating():
    sim = run("decentralized", robot_failures={1: 10.0})
    late = [m for m in sim.comm.log if m.t > 10.0]
    assert late and not any(m.delivered and 1 in (m.sender, m.receiver) for m in late)
