import numpy as np
import pytest

from sim.comms import CommunicationLayer
from sim.robot import Robot


def robots_at(*positions, comm_range=6.0):
    return [Robot(id=i, position=np.array(p, dtype=float), comm_range=comm_range) for i, p in enumerate(positions)]


def test_edges_only_within_range_and_between_alive_robots():
    robots = robots_at((0, 0), (5, 0), (12, 0))
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    g = comm.build_graph(robots, 0.0)
    assert set(map(frozenset, g.edges)) == {frozenset({0, 1})}
    robots[1].failed = True
    g = comm.build_graph(robots, 1.0)
    assert g.number_of_edges() == 0 and 1 not in g


def test_link_uses_the_smaller_of_two_ranges():
    robots = robots_at((0, 0), (5, 0))
    robots[1].comm_range = 4.0
    assert CommunicationLayer(rng=np.random.default_rng(0)).build_graph(robots, 0.0).number_of_edges() == 0


def test_delivery_copies_the_payload_and_logs_it():
    a, b = robots_at((0, 0), (3, 0))
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    comm.build_graph([a, b], 2.0)
    payload = {"mean": np.array([1.0, 2.0])}
    assert comm.send_message(a, b, payload)
    payload["mean"][0] = 99.0  # the sender changing its array must not change what was delivered
    assert b.inbox[0]["mean"][0] == 1.0
    record = comm.log[-1]
    assert (record.t, record.sender, record.receiver, record.delivered, record.reason) == (2.0, 0, 1, True, "delivered")
    assert record.payload_size == 16


def test_no_link_means_no_delivery():
    a, b = robots_at((0, 0), (10, 0))
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    comm.build_graph([a, b], 0.0)
    assert not comm.send_message(a, b, {"x": 1})
    assert b.inbox == [] and comm.log[-1].reason == "no_link"


def test_broadcast_reaches_neighbours_only():
    robots = robots_at((0, 0), (3, 0), (5, 0), (20, 0))
    comm = CommunicationLayer(rng=np.random.default_rng(0))
    comm.build_graph(robots, 0.0)
    assert comm.broadcast(robots[0], robots, {"x": 1}) == 2
    assert [len(r.inbox) for r in robots] == [0, 1, 1, 0]


def test_packet_loss_rate():
    a, b = robots_at((0, 0), (1, 0))
    comm = CommunicationLayer(packet_loss_prob=0.3, rng=np.random.default_rng(1))
    comm.build_graph([a, b], 0.0)
    delivered = sum(comm.send_message(a, b, {"x": 1}) for _ in range(5000))
    assert delivered / 5000 == pytest.approx(0.7, abs=0.03)
    assert comm.messages_lost() == 5000 - delivered


def test_sending_does_not_change_the_simulation_random_stream():
    """Fair comparison: a strategy that sends many messages must see the same world."""
    def next_world_draw(n_messages):
        world_rng = np.random.default_rng(42)
        a, b = robots_at((0, 0), (1, 0))
        comm = CommunicationLayer(packet_loss_prob=0.5, rng=world_rng)
        comm.build_graph([a, b], 0.0)
        for _ in range(n_messages):
            comm.send_message(a, b, {"x": 1})
        return world_rng.random()

    assert next_world_draw(0) == next_world_draw(1000)
