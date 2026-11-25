"""Per-request records -> run summaries.

Only arrivals in [warmup_s, duration_s) count. The measured duration is
duration_s - warmup_s. Requests that arrive in the window but resolve after duration_s
(at most timeout_s later) are still counted, so no arrival is silently dropped.
Outage probability = outages / offered; no_path requests are counted separately (they
cannot occur on a connected topology).
Key utilization uses per-link counter snapshots at warmup_s and duration_s (same window).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qkdsim.config import Config
from qkdsim.simulator import SimulationResult


def measured_records(result: SimulationResult, cfg: Config) -> pd.DataFrame:
    sim = cfg.simulation
    r = result.records
    return r[(r["arrival_s"] >= sim.warmup_s) & (r["arrival_s"] < sim.duration_s)]


def summarize(result: SimulationResult, cfg: Config) -> dict[str, Any]:
    sim = cfg.simulation
    k_bits = cfg.traffic.key_size_bits
    measured_s = sim.duration_s - sim.warmup_s
    r = measured_records(result, cfg)
    delivered = r[r["outcome"] == "delivered"]
    n_offered = len(r)
    n_delivered = len(delivered)
    n_outage = int((r["outcome"] == "outage").sum())
    n_no_path = int((r["outcome"] == "no_path").sum())

    lat = delivered["latency_s"].to_numpy()
    ts = result.timeseries
    ts = ts[ts["time_s"] >= sim.warmup_s]

    def stat(fn, values: np.ndarray) -> float:
        return float(fn(values)) if len(values) else float("nan")

    links = result.links
    produced_bits = float(links["window_produced_bits"].sum())
    link_util = links["window_utilization"].to_numpy()

    return {
        "seed": result.seed,
        "topology_seed": result.topology_seed,
        "strategy": cfg.routing.strategy,
        "topology": cfg.topology.type,
        "n_nodes": result.info["n_nodes"],
        "n_links": result.info["n_links"],
        "avg_degree": result.info["avg_degree"],
        "repair_edges": result.info["repair_edges"],
        "offered_load": result.offered_load,
        "lambda_per_s": result.lambda_per_s,
        "mean_spr_hops": result.mean_spr_hops,
        "rho_bn": result.rho_bn,
        "rho_sat": result.rho_sat,
        "n_bridges": result.load.n_bridges,
        "max_bridge_share": result.load.max_bridge_share,
        "bottleneck_is_bridge": result.load.bottleneck_is_bridge,
        "total_rate_bps": result.total_rate_bps,
        "measured_duration_s": measured_s,
        "offered": n_offered,
        "delivered": n_delivered,
        "outage": n_outage,
        "no_path": n_no_path,
        "outage_probability": n_outage / n_offered if n_offered else float("nan"),
        "delivered_throughput_bps": n_delivered * k_bits / measured_s,
        "latency_mean_s": stat(np.mean, lat),
        "latency_median_s": stat(np.median, lat),
        "latency_p95_s": stat(lambda x: np.percentile(x, 95), lat),
        "wait_mean_s": stat(np.mean, delivered["wait_s"].to_numpy()),
        "mean_hops": stat(np.mean, r.loc[r["outcome"] != "no_path", "hops"].to_numpy()),
        "mean_attempts": stat(np.mean, r["n_attempts"].to_numpy()),
        "mean_fill_fraction": stat(np.mean, ts["mean_fill_fraction"].to_numpy()),
        "frac_links_below_k": stat(np.mean, ts["frac_links_below_k"].to_numpy()),
        # Key utilization over [warmup_s, duration_s): consumed / produced (= generated +
        # discarded), network-wide and per link.
        "key_utilization": float(links["window_consumed_bits"].sum()) / produced_bits
        if produced_bits > 0 else float("nan"),
        "link_utilization_min": stat(np.nanmin, link_util),
        "link_utilization_median": stat(np.nanmedian, link_util),
        "link_utilization_max": stat(np.nanmax, link_util),
    }


def by_hops(result: SimulationResult, cfg: Config) -> pd.DataFrame:
    """Per hop count: number of requests, outage probability, mean latency (delivered)."""
    r = measured_records(result, cfg)
    r = r[r["outcome"] != "no_path"]
    rows = []
    for hops, group in r.groupby("hops", sort=True):
        delivered = group[group["outcome"] == "delivered"]
        rows.append({
            "hops": int(hops),
            "count": len(group),
            "outage_probability": float((group["outcome"] == "outage").mean()),
            "latency_mean_s": float(delivered["latency_s"].mean()) if len(delivered) else float("nan"),
        })
    return pd.DataFrame(rows, columns=["hops", "count", "outage_probability", "latency_mean_s"])
