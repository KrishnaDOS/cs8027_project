"""SimPy discrete-event engine for trusted-node QKD key relay.

Assumptions (see PLAN.md):
  A1 lazy key-pool refill (links.KeyLink)
  A2 OTP relay: delivering K bits over path P consumes K bits on EVERY link of P
  A3 atomic consumption: all links on the path must hold >= K at the same instant
  A4 otherwise wait and retry until timeout_s, then OUTAGE
  A5 waiting requests are not served FIFO (whoever retries first after key arrives wins)
  A6 latency = wait + hops * per_hop_processing_s + path_length_km * propagation_s_per_km

Run lifecycle: arrivals are generated in [0, duration_s). The simulation then keeps
running until every in-flight request has resolved (at most timeout_s after its arrival),
so no request is left unresolved. Metrics later keep arrivals in [warmup_s, duration_s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import simpy

from qkdsim.config import Config, ConfigError
from qkdsim.links import KeyLink, attach_links
from qkdsim.routing import Router, make_router
from qkdsim.topologies import build_topology
from qkdsim.traffic import PairSampler, make_rng_streams, next_interarrival_s, resolve_arrival_rate

log = logging.getLogger(__name__)

OUTCOMES = ("delivered", "outage", "no_path")

RECORD_COLUMNS = [
    "request_id", "arrival_s", "src", "dst", "outcome", "path", "hops", "path_length_km",
    "wait_s", "latency_s", "n_attempts", "resolved_s",
]


@dataclass
class SimulationResult:
    """Everything one run produces. `records` holds ALL arrivals (warmup included)."""

    records: pd.DataFrame
    timeseries: pd.DataFrame
    links: pd.DataFrame
    seed: int
    lambda_per_s: float
    offered_load: float
    mean_spr_hops: float
    total_rate_bps: float
    end_time_s: float
    n_spawned: int
    info: dict[str, Any] = field(default_factory=dict)


def validate_for_run(cfg: Config) -> None:
    """Checks that only matter once a simulation starts."""
    k = cfg.traffic.key_size_bits
    cap = cfg.links.pool_capacity_bits
    if k > cap:
        raise ConfigError(
            f"traffic.key_size_bits={k} exceeds links.pool_capacity_bits={cap}: no link could "
            "ever hold one request's key (time_until would be infinite). Lower key_size_bits "
            "or raise pool_capacity_bits."
        )


def build_network(cfg: Config, topology_rng: np.random.Generator) -> nx.Graph:
    G = build_topology(cfg.topology, topology_rng)
    return attach_links(G, cfg.links)


class _Engine:
    def __init__(self, cfg: Config, G: nx.Graph, router: Router, traffic_rng: np.random.Generator,
                 lambda_per_s: float) -> None:
        self.cfg = cfg
        self.G = G
        self.router = router
        self.rng = traffic_rng
        self.lambda_per_s = lambda_per_s
        self.pairs = PairSampler(G, cfg.traffic, traffic_rng)
        self.env = simpy.Environment()
        self.links: list[KeyLink] = [link for _, _, link in G.edges(data="link")]
        self._path_links: dict[tuple, tuple[list[KeyLink], float]] = {}
        self.records: list[dict[str, Any]] = []
        self.samples: list[dict[str, float]] = []
        self.n_spawned = 0

    # ----------------------------------------------------------------- helpers

    def _links_on(self, path: list[Any]) -> tuple[list[KeyLink], float]:
        """KeyLinks along `path` and its total length (cached per path)."""
        key = tuple(path)
        cached = self._path_links.get(key)
        if cached is None:
            links = [self.G.edges[u, v]["link"] for u, v in zip(path[:-1], path[1:])]
            cached = (links, float(sum(link.length_km for link in links)))
            self._path_links[key] = cached
        return cached

    # ----------------------------------------------------------------- processes

    def arrivals(self):
        duration_s = self.cfg.simulation.duration_s
        while True:
            dt = next_interarrival_s(self.rng, self.lambda_per_s)
            if self.env.now + dt >= duration_s:
                return  # no arrivals at or after duration_s
            yield self.env.timeout(dt)
            src, dst = self.pairs.sample()
            self.env.process(self.handle_request(self.n_spawned, src, dst))
            self.n_spawned += 1

    def handle_request(self, request_id: int, src: Any, dst: Any):
        env = self.env
        k_bits = self.cfg.traffic.key_size_bits
        relay = self.cfg.relay
        arrival_s = env.now
        deadline_s = arrival_s + relay.timeout_s
        n_attempts = 0
        final_check = False

        while True:
            n_attempts += 1
            path = self.router.select_path(self.G, src, dst, env.now)  # re-selected every retry
            if path is None:
                self._record(request_id, arrival_s, src, dst, "no_path", None, 0, 0.0, n_attempts)
                return
            links, length_km = self._links_on(path)
            now = env.now
            # One pass: time_until(now, K) is 0 exactly when the link holds >= K bits,
            # so the longest wait is 0 iff every link on the path can supply K now.
            longest_wait_s = 0.0
            for link in links:
                link_wait_s = link.time_until(now, k_bits)
                if link_wait_s > longest_wait_s:
                    longest_wait_s = link_wait_s
            if longest_wait_s == 0.0:
                for link in links:  # A3: debit every link at the same instant (no yield)
                    link.consume(now, k_bits)
                self._record(request_id, arrival_s, src, dst, "delivered", path, len(links),
                             length_km, n_attempts)
                return
            if final_check:
                self._record(request_id, arrival_s, src, dst, "outage", path, len(links),
                             length_km, n_attempts)
                return
            wait_s = max(relay.min_retry_interval_s, longest_wait_s)
            if now + wait_s >= deadline_s:
                wait_s = max(0.0, deadline_s - now)  # wait until the deadline, check once more
                final_check = True
            yield env.timeout(wait_s)

    def _record(self, request_id: int, arrival_s: float, src: Any, dst: Any, outcome: str,
                path: list[Any] | None, hops: int, length_km: float, n_attempts: int) -> None:
        relay = self.cfg.relay
        now = self.env.now
        wait_s = now - arrival_s
        if outcome == "delivered":
            latency_s = (wait_s + hops * relay.per_hop_processing_s
                         + length_km * relay.propagation_s_per_km)  # A6
        else:
            latency_s = float("nan")
        self.records.append({
            "request_id": request_id, "arrival_s": arrival_s, "src": src, "dst": dst,
            "outcome": outcome, "path": tuple(path) if path is not None else None,
            "hops": hops, "path_length_km": length_km, "wait_s": wait_s,
            "latency_s": latency_s, "n_attempts": n_attempts, "resolved_s": now,
        })

    def sampler(self):
        sim = self.cfg.simulation
        k_bits = self.cfg.traffic.key_size_bits
        n_links = len(self.links)
        while self.env.now < sim.duration_s:
            now = self.env.now
            levels = [link.available(now) for link in self.links]
            fill = sum(lv / link.capacity_bits for lv, link in zip(levels, self.links))
            below = sum(1 for lv in levels if lv < k_bits)
            self.samples.append({
                "time_s": now,
                "mean_fill_fraction": fill / n_links,
                "frac_links_below_k": below / n_links,
            })
            yield self.env.timeout(sim.sample_interval_s)

    # ----------------------------------------------------------------- run

    def run(self) -> float:
        self.env.process(self.arrivals())
        self.env.process(self.sampler())
        self.env.run()  # no `until`: runs until every process (incl. in-flight requests) ends
        if len(self.records) != self.n_spawned:
            raise RuntimeError(
                f"{self.n_spawned - len(self.records)} request(s) unresolved at end of run"
            )
        return self.env.now

    def links_snapshot(self, end_time_s: float) -> pd.DataFrame:
        rows = []
        for u, v, link in self.G.edges(data="link"):
            pool_bits = link.available(end_time_s)
            rows.append({
                "u": u, "v": v, "length_km": link.length_km, "rate_bps": link.rate_bps,
                "capacity_bits": link.capacity_bits, "initial_bits": link.initial_bits,
                "final_bits": pool_bits, "generated_bits": link.total_generated_bits,
                "consumed_bits": link.total_consumed_bits,
                "discarded_bits": link.total_discarded_bits,
            })
        return pd.DataFrame(rows)


def run_simulation(cfg: Config, seed: int) -> SimulationResult:
    """Run one simulation of `cfg` with `seed`. Deterministic for a given (cfg, seed)."""
    cfg.validate()
    validate_for_run(cfg)
    topology_rng, traffic_rng = make_rng_streams(seed)
    G = build_network(cfg, topology_rng)
    lambda_per_s, rho, hops = resolve_arrival_rate(G, cfg.traffic)
    log.info("seed=%d topology=%s nodes=%d links=%d E[h_spr]=%.3f rho=%.3f lambda=%.3f/s",
             seed, cfg.topology.type, G.number_of_nodes(), G.number_of_edges(), hops, rho,
             lambda_per_s)

    engine = _Engine(cfg, G, make_router(cfg.routing), traffic_rng, lambda_per_s)
    end_time_s = engine.run()

    records = pd.DataFrame(engine.records, columns=RECORD_COLUMNS)
    records = records.sort_values("request_id", ignore_index=True)
    return SimulationResult(
        records=records,
        timeseries=pd.DataFrame(engine.samples,
                                columns=["time_s", "mean_fill_fraction", "frac_links_below_k"]),
        links=engine.links_snapshot(end_time_s),
        seed=seed,
        lambda_per_s=lambda_per_s,
        offered_load=rho,
        mean_spr_hops=hops,
        total_rate_bps=float(sum(link.rate_bps for link in engine.links)),
        end_time_s=end_time_s,
        n_spawned=engine.n_spawned,
        info={
            "n_nodes": G.number_of_nodes(),
            "n_links": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
            "repair_edges": G.graph.get("repair_edges", 0),
        },
    )
