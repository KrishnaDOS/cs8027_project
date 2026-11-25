"""Tests for RNG streams, load normalization and pair selection."""

from collections import Counter

import networkx as nx
import numpy as np
import pytest

from qkdsim.config import ConfigError, LinksConfig, TrafficConfig
from qkdsim.links import attach_links
from qkdsim.topologies import make_chain
from qkdsim.traffic import (
    PairSampler,
    make_rng_streams,
    mean_spr_hops,
    next_interarrival_s,
    resolve_arrival_rate,
    run_rng_streams,
    spr_edge_shares,
    traffic_pairs,
)


def test_rng_streams_deterministic_and_independent() -> None:
    topo_a, traffic_a = make_rng_streams(7)
    topo_b, traffic_b = make_rng_streams(7)
    assert topo_a.random() == topo_b.random()
    assert traffic_a.random() == traffic_b.random()
    topo, traffic = make_rng_streams(7)
    assert topo.random() != traffic.random()
    assert make_rng_streams(8)[1].random() != make_rng_streams(7)[1].random()


def test_run_rng_streams_topology_seed() -> None:
    # Unset: topology stream follows the run seed.
    t5, r5, used5 = run_rng_streams(5, None)
    assert used5 == 5
    # Fixed topology seed: same topology stream for every run seed; traffic still varies.
    ta, ra, used_a = run_rng_streams(1, 0)
    tb, rb, used_b = run_rng_streams(2, 0)
    assert used_a == used_b == 0
    assert ta.random() == tb.random()
    assert ra.random() != rb.random()
    # topology.seed=s reproduces the topology that run seed s draws when unset.
    assert run_rng_streams(9, 5)[0].random() == run_rng_streams(5, None)[0].random()
    # The traffic stream depends only on the run seed.
    assert run_rng_streams(5, 0)[1].random() == run_rng_streams(5, None)[1].random()


def test_mean_spr_hops_chain() -> None:
    # Chain of 4: unordered pair distances 1,2,3,1,2,1 -> mean 10/6 (same for ordered pairs).
    assert mean_spr_hops(make_chain(4, 10.0)) == pytest.approx(10 / 6)


def test_mean_spr_hops_rejects_disconnected() -> None:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="disconnected"):
        mean_spr_hops(G)


def test_lambda_from_rho() -> None:
    G = attach_links(make_chain(4, 10.0), LinksConfig())  # 3 links x 5000 bps
    load = resolve_arrival_rate(G, TrafficConfig(key_size_bits=256, offered_load=0.5))
    assert load.offered_load == 0.5
    assert load.mean_spr_hops == pytest.approx(10 / 6)
    assert load.lambda_per_s == pytest.approx(0.5 * 15000 / (256 * 10 / 6))


def test_rho_from_lambda() -> None:
    G = attach_links(make_chain(4, 10.0), LinksConfig())
    cfg = TrafficConfig(key_size_bits=256, offered_load=None, arrival_rate_per_s=10.0)
    load = resolve_arrival_rate(G, cfg)
    assert load.lambda_per_s == 10.0
    assert load.offered_load == pytest.approx(10.0 * 256 * (10 / 6) / 15000)


def test_bottleneck_diagnostic_on_chain() -> None:
    # Chain 0-1-2-3: the middle edge (1,2) is used by 8 of the 12 ordered pairs; the outer
    # edges by 6. So b_max = 8/12 and rho_bn = lambda*K*(2/3)/R.
    G = attach_links(make_chain(4, 10.0), LinksConfig())
    load = resolve_arrival_rate(G, TrafficConfig(key_size_bits=256, offered_load=0.5))
    assert load.bottleneck_edge == (1, 2)
    assert load.bottleneck_share == pytest.approx(8 / 12)
    assert load.rho_bn == pytest.approx(load.lambda_per_s * 256 * (8 / 12) / 5000)
    assert load.rho_sat == pytest.approx(load.offered_load / load.rho_bn)
    # rho_bn scales with lambda; rho_sat does not depend on the load at all.
    load2 = resolve_arrival_rate(G, TrafficConfig(key_size_bits=256, offered_load=1.0))
    assert load2.rho_bn == pytest.approx(2 * load.rho_bn)
    assert load2.rho_sat == pytest.approx(load.rho_sat)


def test_bridge_diagnostics() -> None:
    # Chain: every edge is a bridge; the busiest (1,2) carries 8/12 of ordered pairs.
    G = attach_links(make_chain(4, 10.0), LinksConfig())
    load = resolve_arrival_rate(G, TrafficConfig())
    assert load.n_bridges == 3
    assert load.max_bridge_share == pytest.approx(8 / 12)
    assert load.bottleneck_is_bridge
    # Ring of 6 plus a pendant node 6 hanging off node 0: one bridge, (0, 6).
    R = nx.cycle_graph(6)
    R.add_edge(0, 6)
    nx.set_edge_attributes(R, 10.0, "length_km")
    R = attach_links(R, LinksConfig())
    load = resolve_arrival_rate(R, TrafficConfig())
    assert load.n_bridges == 1
    assert load.max_bridge_share == pytest.approx(2 * 1 * 6 / (7 * 6))  # pairs touching node 6
    # No bridges at all -> share 0.
    C = nx.cycle_graph(5)
    nx.set_edge_attributes(C, 10.0, "length_km")
    load = resolve_arrival_rate(attach_links(C, LinksConfig()), TrafficConfig())
    assert (load.n_bridges, load.max_bridge_share, load.bottleneck_is_bridge) == (0, 0.0, False)


def test_spr_edge_shares_sum_to_mean_hops() -> None:
    G = attach_links(make_chain(5, 10.0), LinksConfig())
    pairs = traffic_pairs(G, TrafficConfig())
    shares = spr_edge_shares(G, pairs)
    assert sum(shares.values()) == pytest.approx(mean_spr_hops(G))


def test_fixed_pair_rho_normalized_by_path_bottleneck() -> None:
    # Distance model on a chain with uneven spacing: rates differ per link.
    from qkdsim.config import DistanceRateConfig
    G = make_chain(4, 25.0)
    G.edges[1, 2]["length_km"] = 75.0  # 10 dB more loss -> 500 bps on the middle link
    G = attach_links(G, LinksConfig(rate_model="distance", distance=DistanceRateConfig()))
    cfg = TrafficConfig(pair_mode="fixed", fixed_pair=[0, 3], key_size_bits=256, offered_load=1.0)
    load = resolve_arrival_rate(G, cfg)
    assert load.lambda_per_s == pytest.approx(500 / 256)  # bottleneck rate / K
    assert load.mean_spr_hops == 3
    assert load.rho_bn == pytest.approx(1.0)  # rho = 1 saturates the bottleneck exactly
    assert load.rho_sat == pytest.approx(1.0)
    assert load.bottleneck_edge == (1, 2)
    # A pair that avoids the slow link is normalized by its own (faster) bottleneck.
    cfg01 = TrafficConfig(pair_mode="fixed", fixed_pair=[0, 1], key_size_bits=256, offered_load=1.0)
    assert resolve_arrival_rate(G, cfg01).lambda_per_s == pytest.approx(5000 / 256)


def test_uniform_pairs_distinct_and_cover_all_ordered_pairs() -> None:
    G = make_chain(4, 10.0)
    sampler = PairSampler(G, TrafficConfig(), np.random.default_rng(0))
    counts = Counter(sampler.sample() for _ in range(12000))
    assert all(s != d for s, d in counts)
    assert len(counts) == 12  # 4 * 3 ordered pairs
    assert max(counts.values()) / min(counts.values()) < 1.15  # ~1000 each


def test_fixed_pair_mode() -> None:
    G = make_chain(4, 10.0)
    cfg = TrafficConfig(pair_mode="fixed", fixed_pair=[0, 3])
    sampler = PairSampler(G, cfg, np.random.default_rng(0))
    assert {sampler.sample() for _ in range(10)} == {(0, 3)}
    with pytest.raises(ConfigError, match="not in the topology"):
        PairSampler(G, TrafficConfig(pair_mode="fixed", fixed_pair=[0, 9]), np.random.default_rng(0))


def test_interarrival_mean() -> None:
    rng = np.random.default_rng(1)
    gaps = [next_interarrival_s(rng, 20.0) for _ in range(20000)]
    assert np.mean(gaps) == pytest.approx(1 / 20.0, rel=0.03)
