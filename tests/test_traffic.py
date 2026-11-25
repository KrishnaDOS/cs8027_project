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
)


def test_rng_streams_deterministic_and_independent() -> None:
    topo_a, traffic_a = make_rng_streams(7)
    topo_b, traffic_b = make_rng_streams(7)
    assert topo_a.random() == topo_b.random()
    assert traffic_a.random() == traffic_b.random()
    topo, traffic = make_rng_streams(7)
    assert topo.random() != traffic.random()
    assert make_rng_streams(8)[1].random() != make_rng_streams(7)[1].random()


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
    lam, rho, hops = resolve_arrival_rate(G, TrafficConfig(key_size_bits=256, offered_load=0.5))
    assert rho == 0.5
    assert hops == pytest.approx(10 / 6)
    assert lam == pytest.approx(0.5 * 15000 / (256 * 10 / 6))


def test_rho_from_lambda() -> None:
    G = attach_links(make_chain(4, 10.0), LinksConfig())
    cfg = TrafficConfig(key_size_bits=256, offered_load=None, arrival_rate_per_s=10.0)
    lam, rho, _ = resolve_arrival_rate(G, cfg)
    assert lam == 10.0
    assert rho == pytest.approx(10.0 * 256 * (10 / 6) / 15000)


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
