"""Tests for SPR routing and the router registry (KAR arrives in M3)."""

import networkx as nx

from qkdsim.config import RoutingConfig
from qkdsim.routing import ROUTERS, SPRRouter, make_router


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
    assert "spr" in ROUTERS
    assert isinstance(make_router(RoutingConfig(strategy="spr")), SPRRouter)
