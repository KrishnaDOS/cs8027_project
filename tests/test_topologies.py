"""Tests for topology builders and per_link rate handling."""

import hashlib
import warnings

import networkx as nx
import numpy as np
import pytest
import yaml

from qkdsim.config import ConfigError, LinksConfig
from qkdsim.links import attach_links
from qkdsim.topologies import (
    SECOQC_PATH,
    AssumedRateWarning,
    make_chain,
    make_random_mesh,
    make_secoqc,
    mesh_radius_km,
)

ignore_assumed = pytest.mark.filterwarnings("ignore::qkdsim.topologies.AssumedRateWarning")


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def secoqc_file_links() -> list[dict]:
    return yaml.safe_load(SECOQC_PATH.read_text(encoding="utf-8"))["links"]


# --------------------------------------------------------------------------- chain


def test_chain_defaults() -> None:
    G = make_chain()
    assert G.number_of_nodes() == 32
    assert G.number_of_edges() == 31
    assert all(length == 63.8 for _, _, length in G.edges(data="length_km"))
    assert nx.is_connected(G)
    assert nx.shortest_path_length(G, 0, 31) == 31


def test_chain_custom_spacing() -> None:
    G = make_chain(4, spacing_km=10.0)
    assert sorted(G.edges()) == [(0, 1), (1, 2), (2, 3)]
    assert G.nodes[3]["pos"] == (30.0, 0.0)


# --------------------------------------------------------------------------- SECOQC


@ignore_assumed
def test_secoqc_shape_and_lengths() -> None:
    G = make_secoqc()
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 8
    assert set(G.nodes) == {"STP", "BRT", "SIE", "ERD", "GUD", "FRM"}
    for link in secoqc_file_links():
        assert G.edges[link["u"], link["v"]]["length_km"] == link["length_km"]
    assert nx.is_connected(G)


@ignore_assumed
def test_secoqc_keeps_extra_fields() -> None:
    G = make_secoqc()
    edge = G.edges["BRT", "SIE"]
    assert edge["system"] == "Toshiba"
    assert edge["skr_status"] == "measured"
    assert "note" in edge
    assert G.nodes["STP"]["name"] == "St. Poelten repeater station"


def test_secoqc_warns_about_assumed_links() -> None:
    with pytest.warns(AssumedRateWarning) as record:
        make_secoqc()
    assert len(record) == 1
    message = str(record[0].message)
    assert "idQ-2" in message
    assert "idQ-3" in message
    assert "2 link(s)" in message
    for system in ("COW", "Toshiba", "Entangled", "FreeSpace", "CV", "idQ-1"):
        assert f"({system}," not in message


@ignore_assumed
def test_secoqc_per_link_rates_match_file() -> None:
    G = attach_links(make_secoqc(), LinksConfig(rate_model="per_link"))
    for link in secoqc_file_links():
        assert G.edges[link["u"], link["v"]]["link"].rate_bps == link["skr_bps"]


@ignore_assumed
def test_secoqc_file_unchanged_by_loading() -> None:
    before = sha256(SECOQC_PATH)
    mtime_before = SECOQC_PATH.stat().st_mtime_ns
    G = make_secoqc()
    attach_links(G, LinksConfig(rate_model="per_link"))
    assert sha256(SECOQC_PATH) == before
    assert SECOQC_PATH.stat().st_mtime_ns == mtime_before


# --------------------------------------------------------------------------- random mesh


def test_mesh_connected_lengths_and_positions() -> None:
    G = make_random_mesh(30, 50.0, 4.0, np.random.default_rng(0))
    assert G.number_of_nodes() == 30
    assert nx.is_connected(G)
    radius_km = G.graph["radius_km"]
    for u, v, length in G.edges(data="length_km"):
        (x1, y1), (x2, y2) = G.nodes[u]["pos"], G.nodes[v]["pos"]
        assert length == pytest.approx(np.hypot(x1 - x2, y1 - y2))
        assert length <= radius_km
    for _, (x, y) in G.nodes(data="pos"):
        assert 0 <= x <= 50 and 0 <= y <= 50


@pytest.mark.parametrize(("n_nodes", "target"), [(30, 4.0), (20, 4.0), (30, 6.0)])
def test_mesh_average_degree_near_target(n_nodes: int, target: float) -> None:
    seeds = range(10)
    degrees = [
        2 * make_random_mesh(n_nodes, 50.0, target, np.random.default_rng(s)).number_of_edges() / n_nodes
        for s in seeds
    ]
    assert abs(np.mean(degrees) - target) <= 0.15 * target


def test_mesh_is_deterministic_per_seed() -> None:
    a = make_random_mesh(30, 50.0, 4.0, np.random.default_rng(42))
    b = make_random_mesh(30, 50.0, 4.0, np.random.default_rng(42))
    assert sorted(a.edges(data="length_km")) == sorted(b.edges(data="length_km"))


def test_mesh_radius_matches_expected_degree() -> None:
    # Before conditioning on connectivity, the expected degree equals the target.
    rng = np.random.default_rng(1)
    n, area, target = 30, 50.0, 4.0
    r = mesh_radius_km(n, area, target)
    degrees = []
    for _ in range(500):
        pos = rng.uniform(0, area, size=(n, 2))
        dist = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
        degrees.append(((dist <= r).sum() - n) / n)
    assert np.mean(degrees) == pytest.approx(target, rel=0.03)


def test_mesh_fails_loudly_when_never_connected() -> None:
    with pytest.raises(RuntimeError, match="100 tries"):
        make_random_mesh(60, 50.0, 1.0, np.random.default_rng(0))


def test_mesh_rejects_unreachable_degree() -> None:
    with pytest.raises(ValueError, match="not achievable"):
        mesh_radius_km(5, 50.0, 4.5)


# --------------------------------------------------------------------------- per_link errors


def test_per_link_on_chain_raises() -> None:
    with pytest.raises(ConfigError, match="per_link"):
        attach_links(make_chain(), LinksConfig(rate_model="per_link"))


def test_per_link_on_mesh_raises() -> None:
    G = make_random_mesh(30, 50.0, 4.0, np.random.default_rng(0))
    with pytest.raises(ConfigError, match="per_link"):
        attach_links(G, LinksConfig(rate_model="per_link"))
    assert all("link" not in d for _, _, d in G.edges(data=True))  # nothing half-attached


def test_secoqc_with_fixed_rate_model_works() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumedRateWarning)
        G = attach_links(make_secoqc(), LinksConfig(rate_model="fixed"))
    assert all(d["link"].rate_bps == 5000 for _, _, d in G.edges(data=True))
