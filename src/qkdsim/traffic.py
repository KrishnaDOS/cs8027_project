"""Traffic: RNG streams, Poisson arrivals, (src, dst) pair selection, load normalization.

No global RNG: every random draw comes from an explicit np.random.Generator. A seed yields
two independent streams (topology, traffic) via np.random.SeedSequence.spawn, so changing
the router never changes the topology or the arrival sequence for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import networkx as nx
import numpy as np

from qkdsim.config import ConfigError, TrafficConfig
from qkdsim.routing import SPRRouter


def make_rng_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Return (topology_rng, traffic_rng), two independent streams derived from `seed`."""
    topology_ss, traffic_ss = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(topology_ss), np.random.default_rng(traffic_ss)


def run_rng_streams(
    run_seed: int, topology_seed: int | None
) -> tuple[np.random.Generator, np.random.Generator, int]:
    """(topology_rng, traffic_rng, topology_seed_used) for one run.

    topology_seed None: both streams come from run_seed (topology varies with the run seed).
    Otherwise the topology stream comes from topology_seed (fixed topology) and the traffic
    stream from run_seed. For a given seed s, topology_seed=s and topology_seed=None give
    the same topology, so fixing the topology seed never changes which graph seed s means.
    """
    traffic_rng = make_rng_streams(run_seed)[1]
    used = run_seed if topology_seed is None else topology_seed
    return make_rng_streams(used)[0], traffic_rng, used


def mean_spr_hops(G: nx.Graph) -> float:
    """Mean shortest-path hop count over all ordered pairs (s, d), s != d.

    Raises if any pair is disconnected (the load normalization would be undefined).
    """
    n = G.number_of_nodes()
    total_hops = 0
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        if len(lengths) != n:
            raise ValueError(f"graph is disconnected: node {src!r} reaches {len(lengths) - 1} of {n - 1}")
        total_hops += sum(lengths.values())
    return total_hops / (n * (n - 1))


def total_link_rate_bps(G: nx.Graph) -> float:
    """Sum of the KeyLink rates over all edges (requires attach_links)."""
    return float(sum(link.rate_bps for _, _, link in G.edges(data="link")))


@dataclass(frozen=True)
class LoadInfo:
    """Offered load of one run, plus SPR bottleneck diagnostics.

    lambda_per_s, offered_load (rho): linked by lambda = rho * capacity_per_s, where
        uniform: capacity_per_s = sum(R_link) / (K * E[h_spr])   (network-wide)
        fixed:   capacity_per_s = min R_link on the pair's SPR path / K   (bottleneck)
    mean_spr_hops: E[h_spr] over the traffic's pairs (uniform: all ordered pairs;
        fixed: the pair's SPR hop count).
    rho_bn: max_e lambda * K * b_e / R_e, the load factor on SPR's busiest link, where
        b_e is the fraction of the traffic's (s, d) pairs whose SPR path uses edge e.
    rho_sat: rho / rho_bn, the rho at which SPR's busiest link saturates.
    bottleneck_edge, bottleneck_share: argmax edge and its b_e.
    """

    lambda_per_s: float
    offered_load: float
    mean_spr_hops: float
    capacity_per_s: float
    rho_bn: float
    rho_sat: float
    bottleneck_edge: tuple[Any, Any]
    bottleneck_share: float


def spr_edge_shares(G: nx.Graph, pairs: list[tuple[Any, Any]]) -> dict[tuple[Any, Any], float]:
    """b_e: fraction of `pairs` whose SPR path (same tie-breaking as the router) uses e.

    Edges are keyed as (u, v) in G.edges order; edges no path uses are absent.
    """
    router = SPRRouter()
    key_of = {frozenset((u, v)): (u, v) for u, v in G.edges}
    counts: dict[tuple[Any, Any], int] = {}
    for s, d in pairs:
        path = router.select_path(G, s, d, 0.0)
        if path is None:
            raise ValueError(f"no path between {s!r} and {d!r}")
        for a, b in zip(path[:-1], path[1:]):
            e = key_of[frozenset((a, b))]
            counts[e] = counts.get(e, 0) + 1
    return {e: c / len(pairs) for e, c in counts.items()}


def traffic_pairs(G: nx.Graph, cfg: TrafficConfig) -> list[tuple[Any, Any]]:
    """The (s, d) pairs the traffic draws from, each equally likely."""
    if cfg.pair_mode == "fixed":
        src, dst = cfg.fixed_pair
        return [(src, dst)]
    nodes = sorted(G.nodes)
    return [(s, d) for s in nodes for d in nodes if s != d]


def resolve_arrival_rate(G: nx.Graph, cfg: TrafficConfig) -> LoadInfo:
    """Resolve lambda <-> rho for the configured traffic (links must be attached).

    Exactly one of offered_load / arrival_rate_per_s is set; the other is derived.
    uniform: rho = 1 means the network-wide key generation rate equals the key consumed
        by requests on shortest paths. Because SPR concentrates traffic, individual
        links saturate well below rho = 1 (see rho_sat).
    fixed: rho = 1 means the pair's SPR bottleneck link is exactly saturated.
    """
    k_bits = cfg.key_size_bits
    pairs = traffic_pairs(G, cfg)
    for node in {n for pair in pairs for n in pair}:
        if node not in G:
            raise ConfigError(f"traffic.fixed_pair node {node!r} is not in the topology")
    shares = spr_edge_shares(G, pairs)

    if cfg.pair_mode == "fixed":
        hops = float(len(shares))
        capacity_per_s = min(G.edges[e]["link"].rate_bps for e in shares) / k_bits
    else:
        hops = mean_spr_hops(G)
        capacity_per_s = total_link_rate_bps(G) / (k_bits * hops)

    if cfg.offered_load is not None:
        rho = float(cfg.offered_load)
        lam = rho * capacity_per_s
    else:
        lam = float(cfg.arrival_rate_per_s)
        rho = lam / capacity_per_s

    # Busiest SPR link; ties broken by edge order for determinism.
    edge, load = max(
        ((e, lam * k_bits * b / G.edges[e]["link"].rate_bps) for e, b in shares.items()),
        key=lambda item: item[1],
    )
    return LoadInfo(
        lambda_per_s=lam,
        offered_load=rho,
        mean_spr_hops=hops,
        capacity_per_s=capacity_per_s,
        rho_bn=load,
        rho_sat=rho / load,
        bottleneck_edge=edge,
        bottleneck_share=shares[edge],
    )


class PairSampler:
    """Chooses (src, dst) for each request.

    uniform: a uniformly random ordered pair of distinct nodes.
    fixed:   always cfg.fixed_pair.
    Nodes are sorted once so draws do not depend on graph insertion order.
    """

    def __init__(self, G: nx.Graph, cfg: TrafficConfig, rng: np.random.Generator) -> None:
        self.mode = cfg.pair_mode
        self.rng = rng
        self.nodes: list[Hashable] = sorted(G.nodes)
        if self.mode == "fixed":
            src, dst = cfg.fixed_pair  # validated non-None by Config.validate
            for node in (src, dst):
                if node not in G:
                    raise ConfigError(f"traffic.fixed_pair node {node!r} is not in the topology")
            self.fixed_pair: tuple[Any, Any] = (src, dst)
        elif self.mode != "uniform":
            raise ConfigError(f"unknown traffic.pair_mode {self.mode!r}")

    def sample(self) -> tuple[Any, Any]:
        if self.mode == "fixed":
            return self.fixed_pair
        n = len(self.nodes)
        i = int(self.rng.integers(n))
        j = int(self.rng.integers(n - 1))
        if j >= i:
            j += 1
        return self.nodes[i], self.nodes[j]


def next_interarrival_s(rng: np.random.Generator, lambda_per_s: float) -> float:
    """Exponential inter-arrival time of a Poisson process with rate lambda_per_s."""
    return float(rng.exponential(1.0 / lambda_per_s))
