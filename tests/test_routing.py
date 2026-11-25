"""Tests for SPR routing and the router registry (KAR arrives in M3)."""

import networkx as nx
import numpy as np

from qkdsim.config import LinksConfig, RoutingConfig
from qkdsim.links import KeyLink, attach_links
from qkdsim.routing import ROUTERS, KARRouter, SPRRouter, make_router
from qkdsim.topologies import make_random_mesh


def test_spr_picks_fewest_hops() -> None:
    G = nx.Graph()
    nx.add_path(G, [0, 1, 2, 3, 4])  # 4 hops
    G.add_edges_from([(0, 5), (5, 4)])  # 2 hops
    assert SPRRouter().select_path(G, 0, 4, 0.0) == [0, 5, 4]


def test_spr_tie_break_is_lexicographic_and_order_independent() -> None:
    edges = [(0, 2), (2, 3), (0, 1), (1, 3)]  # two 2-hop paths: 0-1-3 and 0-2-3
    for order in (edges, list(reversed(edges))):
        G = nx.Graph()
        G.add_edges_from(order)
        assert SPRRouter().select_path(G, 0, 3, 0.0) == [0, 1, 3]
        assert SPRRouter().select_path(G, 3, 0, 0.0) == [3, 1, 0]


def test_spr_returns_none_without_path() -> None:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    assert SPRRouter().select_path(G, 0, 3, 0.0) is None


def test_spr_cache_resets_for_new_graph() -> None:
    router = SPRRouter()
    G1 = nx.path_graph(3)
    assert router.select_path(G1, 0, 2, 0.0) == [0, 1, 2]
    G2 = nx.path_graph(3)
    G2.add_edge(0, 2)
    assert router.select_path(G2, 0, 2, 0.0) == [0, 2]


def test_registry() -> None:
    assert set(ROUTERS) == {"spr", "kar"}
    assert isinstance(make_router(RoutingConfig(strategy="spr")), SPRRouter)
    assert isinstance(make_router(RoutingConfig(strategy="kar")), KARRouter)


# --------------------------------------------------------------------------- KAR


def linked(edges, capacity_bits: float = 1000.0, initial_bits: float = 1000.0) -> nx.Graph:
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v, link=KeyLink(rate_bps=10.0, capacity_bits=capacity_bits,
                                      initial_bits=initial_bits, length_km=1.0))
    return G


def test_kar_avoids_drained_link_that_spr_uses() -> None:
    G = linked([(0, 3), (0, 1), (1, 2), (2, 3)])  # direct 1-hop edge vs a 3-hop detour
    assert SPRRouter().select_path(G, 0, 3, 0.0) == [0, 3]
    assert KARRouter().select_path(G, 0, 3, 0.0) == [0, 3]  # all full: fewest hops wins
    G.edges[0, 3]["link"].consume(0.0, 1000.0)  # drain the direct link
    assert SPRRouter().select_path(G, 0, 3, 0.0) == [0, 3]  # SPR ignores key levels
    assert KARRouter().select_path(G, 0, 3, 0.0) == [0, 1, 2, 3]


def test_kar_follows_key_levels_over_time() -> None:
    G = linked([(0, 1), (1, 3), (0, 2), (2, 3)])
    router = KARRouter()
    assert router.select_path(G, 0, 3, 0.0) == [0, 1, 3]  # tie -> smaller predecessor
    G.edges[0, 1]["link"].consume(0.0, 990.0)  # 10 bits left on 0-1
    assert router.select_path(G, 0, 3, 0.0) == [0, 2, 3]  # cache invalidated by consume
    # After 99 s at 10 bps, 0-1 is full again: back to the tie-break path.
    assert router.select_path(G, 0, 3, 99.0) == [0, 1, 3]


def test_kar_tie_break_independent_of_insertion_order() -> None:
    edges = [(0, 2), (2, 3), (0, 1), (1, 3), (1, 2)]
    results = set()
    for order in (edges, list(reversed(edges))):
        results.add(tuple(KARRouter().select_path(linked(order), 0, 3, 0.0)))
    assert results == {(0, 1, 3)}


def test_kar_does_not_mutate_link_state() -> None:
    G = linked([(0, 1), (1, 2), (0, 2)], initial_bits=300.0)
    G.edges[0, 1]["link"].consume(1.0, 100.0)
    before = {e: dict(vars(G.edges[e]["link"])) for e in G.edges}
    KARRouter().select_path(G, 0, 2, 5.0)
    KARRouter().select_path(G, 2, 0, 7.5)
    assert {e: dict(vars(G.edges[e]["link"])) for e in G.edges} == before


def test_kar_no_path_and_trivial_path() -> None:
    G = linked([(0, 1), (2, 3)])
    assert KARRouter().select_path(G, 0, 3, 0.0) is None
    assert KARRouter().select_path(G, 0, 0, 0.0) == [0]


def test_kar_paths_use_only_existing_edges() -> None:
    rng = np.random.default_rng(0)
    for seed in range(5):
        G = attach_links(make_random_mesh(30, 50.0, 4.0, np.random.default_rng(seed)), LinksConfig())
        for _, _, link in G.edges(data="link"):  # random drain levels
            link.consume(0.0, float(rng.uniform(0, link.available(0.0))))
        router = KARRouter()
        nodes = sorted(G.nodes)
        for _ in range(200):
            s, d = (int(x) for x in rng.choice(nodes, size=2, replace=False))
            now = float(rng.uniform(0.0, 5.0))
            path = router.select_path(G, s, d, now)
            assert path is not None and path[0] == s and path[-1] == d
            assert len(set(path)) == len(path)  # simple path
            assert all(G.has_edge(a, b) for a, b in zip(path[:-1], path[1:]))


def test_kar_cache_matches_fresh_computation() -> None:
    rng = np.random.default_rng(1)
    G = attach_links(make_random_mesh(20, 50.0, 4.0, np.random.default_rng(2)), LinksConfig())
    cached = KARRouter()
    links = [link for _, _, link in G.edges(data="link")]
    now = 0.0
    for step in range(300):
        now += float(rng.choice([0.0, 0.01]))  # many queries at the same instant
        if rng.random() < 0.3:
            link = links[int(rng.integers(len(links)))]
            link.consume(now, min(256.0, link.available(now)))
        s, d = (int(x) for x in rng.choice(20, size=2, replace=False))
        assert cached.select_path(G, s, d, now) == KARRouter().select_path(G, s, d, now)
