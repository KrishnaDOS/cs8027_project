"""Topology builders.

Each builder returns an nx.Graph whose edges carry "length_km" (and, for SECOQC, the extra
per-link fields from the data file). Nodes may carry "pos" = (x_km, y_km).
KeyLinks are added afterwards with qkdsim.links.attach_links(G, cfg.links).
Terrestrial networks only (assumption A7).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from qkdsim.config import ChainTopologyConfig, ConfigError, TopologyConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SECOQC_PATH = REPO_ROOT / "data" / "secoqc.yaml"


class AssumedRateWarning(UserWarning):
    """A topology data file contains links whose key rate is assumed, not sourced."""


# --------------------------------------------------------------------------- chain


def make_chain(
    n_nodes: int = 32, spacing_km: float = ChainTopologyConfig.spacing_km
) -> nx.Graph:
    """Line of n_nodes trusted nodes 0..n-1, consecutive nodes spacing_km apart.

    Defaults model the Beijing-Shanghai backbone: 32 trusted relays, 31 links averaging
    63.8 km (Chen 2021, Nature, Table 1; Addendum A.1), ~2000 km end to end.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2; got {n_nodes}")
    if spacing_km <= 0:
        raise ValueError(f"spacing_km must be > 0; got {spacing_km}")
    G = nx.Graph(name="chain")
    for i in range(n_nodes):
        G.add_node(i, pos=(i * spacing_km, 0.0))
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1, length_km=spacing_km)
    return G


# --------------------------------------------------------------------------- SECOQC


def make_secoqc(path: str | Path = SECOQC_PATH) -> nx.Graph:
    """SECOQC Vienna network (Peev et al. 2009): 6 nodes, 8 links.

    Loaded READ-ONLY from data/secoqc.yaml (real data, Addendum A.3); this function never
    writes to the file. Every per-link field other than u/v (length_km, system, skr_bps,
    skr_status, note, ...) is kept as an edge attribute. Emits an AssumedRateWarning
    listing the links whose skr_status is "assumed".
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    G = nx.Graph(name=data.get("name", "secoqc"))
    for node in data["nodes"]:
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node["id"], **attrs)

    for link in data["links"]:
        u, v = link["u"], link["v"]
        for end in (u, v):
            if end not in G:
                raise ValueError(f"{path}: link {u}-{v} references unknown node {end!r}")
        if G.has_edge(u, v):
            raise ValueError(f"{path}: duplicate link {u}-{v}")
        if "length_km" not in link:
            raise ValueError(f"{path}: link {u}-{v} has no length_km")
        attrs = {k: val for k, val in link.items() if k not in ("u", "v")}
        G.add_edge(u, v, **attrs)

    assumed = [
        (u, v, d) for u, v, d in G.edges(data=True) if d.get("skr_status") == "assumed"
    ]
    if assumed:
        listed = "; ".join(
            f"{u}-{v} ({d.get('system', '?')}, skr_bps={d.get('skr_bps')})"
            for u, v, d in assumed
        )
        warnings.warn(
            f"SECOQC: {len(assumed)} link(s) use an ASSUMED key rate (no rate given in "
            f"Peev 2009; design criterion used instead): {listed}",
            AssumedRateWarning,
            stacklevel=2,
        )
    return G


# --------------------------------------------------------------------------- random mesh


def _square_pair_probability(d: float) -> float:
    """P(|X - Y| <= d) for X, Y uniform in the unit square, valid for 0 <= d <= 1.

    Exact closed form; it accounts for the boundary (nodes near edges and corners have
    fewer neighbours than pi*r^2 would predict).
    """
    return math.pi * d**2 - (8.0 / 3.0) * d**3 + 0.5 * d**4


def mesh_radius_km(n_nodes: int, area_km: float, target_avg_degree: float) -> float:
    """Connection radius whose EXPECTED average degree equals target_avg_degree.

    Solves (n - 1) * p(r / area_km) = target by bisection, where p is the exact
    boundary-corrected pair probability for a square of side area_km.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2; got {n_nodes}")
    target_p = target_avg_degree / (n_nodes - 1)
    p_max = _square_pair_probability(1.0)  # ~0.975: radius equal to the side length
    if not 0 < target_p <= p_max:
        raise ValueError(
            f"target_avg_degree={target_avg_degree} is not achievable with n_nodes={n_nodes} "
            f"(must be in (0, {p_max * (n_nodes - 1):.2f}])"
        )
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _square_pair_probability(mid) < target_p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * area_km


def make_random_mesh(
    n_nodes: int,
    area_km: float,
    target_avg_degree: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Random geometric graph on n_nodes uniform in an area_km x area_km square, repaired
    to connectivity (Addendum B).

    One draw only: nodes within mesh_radius_km(...) of each other are linked. If that graph
    is disconnected, the shortest Euclidean edge between two different components is
    added, repeatedly, until it is connected. Each repair edge merges two components, so
    G.graph["repair_edges"] == (initial number of components) - 1. Repair edges are longer
    than the radius. Edge length is always the Euclidean distance between the endpoints.
    Unlike reject-and-redraw, this does not bias the sample towards denser draws.
    """
    radius_km = mesh_radius_km(n_nodes, area_km, target_avg_degree)
    pos_km = rng.uniform(0.0, area_km, size=(n_nodes, 2))
    dist_km = np.sqrt(((pos_km[:, None, :] - pos_km[None, :, :]) ** 2).sum(axis=-1))

    G = nx.Graph(name="random_mesh", radius_km=radius_km)
    for i in range(n_nodes):
        G.add_node(i, pos=(float(pos_km[i, 0]), float(pos_km[i, 1])))
    rows, cols = np.nonzero(np.triu(dist_km <= radius_km, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        G.add_edge(i, j, length_km=float(dist_km[i, j]))

    repair_edges = 0
    while not nx.is_connected(G):
        component = np.empty(n_nodes, dtype=int)
        for label, nodes in enumerate(nx.connected_components(G)):
            component[list(nodes)] = label
        # Distances between nodes in different components only; argmin picks the shortest.
        cross_km = np.where(component[:, None] != component[None, :], dist_km, np.inf)
        i, j = np.unravel_index(np.argmin(cross_km), cross_km.shape)
        G.add_edge(int(i), int(j), length_km=float(dist_km[i, j]))
        repair_edges += 1

    G.graph["repair_edges"] = repair_edges
    return G


# --------------------------------------------------------------------------- dispatch


def build_topology(cfg: TopologyConfig, rng: np.random.Generator) -> nx.Graph:
    """Build the topology named by cfg.type (links are NOT attached here).

    chain uses cfg.n_nodes and cfg.chain.spacing_km; secoqc ignores n_nodes (always 6);
    random_mesh uses n_nodes, area_km, target_avg_degree and the topology RNG stream.
    """
    if cfg.type == "chain":
        return make_chain(cfg.n_nodes, cfg.chain.spacing_km)
    if cfg.type == "secoqc":
        return make_secoqc()
    if cfg.type == "random_mesh":
        return make_random_mesh(cfg.n_nodes, cfg.area_km, cfg.target_avg_degree, rng)
    raise ConfigError(f"unknown topology.type {cfg.type!r}")
