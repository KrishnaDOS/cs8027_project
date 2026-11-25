"""Select the "diverse" mesh for E2 (Addendum D).

Scans topology seeds for the default mesh and selects the LOWEST seed whose
max_bridge_share <= threshold (default 0.15). The rule uses topology only (bridges and
SPR pair shares), never routing results, so it can be fixed before any experiment runs.

    python scripts/select_mesh.py [--config configs/default.yaml] [--seeds 50] [--threshold 0.15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from qkdsim.config import Config, load_config
from qkdsim.simulator import build_network
from qkdsim.traffic import resolve_arrival_rate, run_rng_streams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_THRESHOLD = 0.15


def mesh_stats(cfg: Config, topology_seed: int) -> dict[str, Any]:
    """Topology diagnostics for one topology seed (no simulation is run)."""
    G = build_network(cfg, run_rng_streams(0, topology_seed)[0])
    load = resolve_arrival_rate(G, cfg.traffic)
    return {
        "topology_seed": topology_seed,
        "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "repair_edges": G.graph.get("repair_edges", 0),
        "n_bridges": load.n_bridges,
        "max_bridge_share": load.max_bridge_share,
        "bottleneck_is_bridge": load.bottleneck_is_bridge,
        "rho_sat": load.rho_sat,
    }


def scan(cfg: Config, seeds: Sequence[int]) -> list[dict[str, Any]]:
    return [mesh_stats(cfg, s) for s in seeds]


def select_seed(rows: Sequence[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> int | None:
    """Lowest topology seed with max_bridge_share <= threshold (None if none qualifies)."""
    eligible = [r["topology_seed"] for r in rows if r["max_bridge_share"] <= threshold]
    return min(eligible) if eligible else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", type=int, default=50, help="scan topology seeds 0..N-1")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = scan(cfg, range(args.seeds))
    print(f"mesh: n={cfg.topology.n_nodes}, area={cfg.topology.area_km} km, "
          f"target degree={cfg.topology.target_avg_degree}")
    print("seed | avg_deg | repair | n_bridges | max_bridge_share | busiest=bridge | rho_sat")
    for r in rows:
        mark = "  <= threshold" if r["max_bridge_share"] <= args.threshold else ""
        print(f"{r['topology_seed']:4d} | {r['avg_degree']:7.2f} | {r['repair_edges']:6d} | "
              f"{r['n_bridges']:9d} | {r['max_bridge_share']:16.3f} | "
              f"{str(r['bottleneck_is_bridge']):14s} | {r['rho_sat']:.3f}{mark}")
    chosen = select_seed(rows, args.threshold)
    if chosen is None:
        print(f"\nno seed in 0..{args.seeds - 1} has max_bridge_share <= {args.threshold}")
        return 1
    n_ok = sum(r["max_bridge_share"] <= args.threshold for r in rows)
    print(f"\nselected diverse mesh: topology.seed={chosen} "
          f"({n_ok} of {len(rows)} seeds meet max_bridge_share <= {args.threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
