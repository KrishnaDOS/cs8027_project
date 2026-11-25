"""Traffic: RNG streams, Poisson arrivals, (src, dst) pair selection, load normalization.

No global RNG: every random draw comes from an explicit np.random.Generator. A seed yields
two independent streams (topology, traffic) via np.random.SeedSequence.spawn, so changing
the router never changes the topology or the arrival sequence for a given seed.
"""

from __future__ import annotations

from typing import Any, Hashable

import networkx as nx
import numpy as np

from qkdsim.config import ConfigError, TrafficConfig


def make_rng_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Return (topology_rng, traffic_rng), two independent streams derived from `seed`."""
    topology_ss, traffic_ss = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(topology_ss), np.random.default_rng(traffic_ss)


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


def resolve_arrival_rate(G: nx.Graph, cfg: TrafficConfig) -> tuple[float, float, float]:
    """Return (lambda_per_s, rho, mean_spr_hops) for the configured traffic.

    rho and lambda are linked by
        lambda = rho * sum(R_link) / (K * E[h_spr])
    i.e. rho = 1 means the network-wide key generation rate equals the key consumed by
    requests on shortest paths. Exactly one of offered_load / arrival_rate_per_s is set;
    the other is derived. The same normalization is used in "fixed" pair mode.
    """
    hops = mean_spr_hops(G)
    capacity_per_s = total_link_rate_bps(G) / (cfg.key_size_bits * hops)
    if cfg.offered_load is not None:
        rho = float(cfg.offered_load)
        return rho * capacity_per_s, rho, hops
    lam = float(cfg.arrival_rate_per_s)
    return lam, lam / capacity_per_s, hops


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
