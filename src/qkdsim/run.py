"""CLI: one simulation run from a config file.

    python -m qkdsim.run --config configs/default.yaml [--set key=value ...] [--seed N]

--set takes dotted keys; values are parsed as YAML (numbers, lists, null, strings), e.g.
    --set traffic.offered_load=1.0 --set topology.type=chain --set topology.n_nodes=32
Without --seed, the first entry of simulation.seeds is used.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Sequence

import pandas as pd
import yaml

from qkdsim.config import ConfigError, deep_merge, load_config
from qkdsim.metrics import by_hops, summarize
from qkdsim.simulator import run_simulation


def parse_set_args(items: Sequence[str]) -> dict[str, Any]:
    """Turn ["a.b=1", "c=x"] into {"a": {"b": 1}, "c": "x"}."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects key=value, got {item!r}")
        value = yaml.safe_load(raw) if raw.strip() else None
        nested: Any = value
        for part in reversed(key.strip().split(".")):
            nested = {part: nested}
        overrides = deep_merge(overrides, nested)
    return overrides


def format_summary(summary: dict[str, Any]) -> str:
    width = max(len(k) for k in summary)
    lines = []
    for key, value in summary.items():
        shown = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"  {key:<{width}}  {shown}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qkdsim.run", description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value (dotted key; repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="seed (default: first of simulation.seeds)")
    parser.add_argument("--by-hops", action="store_true", help="also print the per-hop-count breakdown")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config, overrides=parse_set_args(args.sets))
    except ConfigError as exc:
        parser.error(str(exc))
    seed = args.seed if args.seed is not None else cfg.simulation.seeds[0]

    t0 = time.perf_counter()
    result = run_simulation(cfg, seed)
    wall_s = time.perf_counter() - t0
    summary = summarize(result, cfg)

    print(f"qkdsim run: {args.config}  seed={seed}  topology={cfg.topology.type}  "
          f"router={cfg.routing.strategy}  rate_model={cfg.links.rate_model}")
    topo_note = "fixed" if cfg.topology.seed is not None else "from run seed"
    print(f"topology seed={result.topology_seed} ({topo_note})")
    print(f"load: rho={result.offered_load:.4g}  lambda={result.lambda_per_s:.4g} req/s  "
          f"(E[h_spr]={result.mean_spr_hops:.4g}, sum R={result.total_rate_bps:.6g} bps, "
          f"K={cfg.traffic.key_size_bits} bits)")
    print(f"SPR bottleneck: rho_bn={result.rho_bn:.4g}  rho_sat={result.rho_sat:.4g}  "
          f"(edge {result.info['bottleneck_edge']}, used by "
          f"{result.info['bottleneck_share']:.1%} of pairs)")
    print(format_summary(summary))
    if args.by_hops:
        with pd.option_context("display.width", 120, "display.max_rows", 200):
            print("\nby hop count:")
            print(by_hops(result, cfg).to_string(index=False))
    print(f"\nwall-clock: {wall_s:.2f} s  (simulated end time {result.end_time_s:.3f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
