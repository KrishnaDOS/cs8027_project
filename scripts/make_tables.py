"""Write figures/tables.md: Markdown tables for E1-E4 and the E2 mesh diagnostics.

    python scripts/make_tables.py [--results-dir results] [--out figures/tables.md]

Values are mean ± 95% CI half-width across seeds (Student's t; qkdsim.analysis).
Experiments without results are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from qkdsim.analysis import (
    MESH_ORDER,
    e3_per_degree,
    group_ci,
    load_results,
    mesh_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def pm(mean: float, ci: float, digits: int = 3, thousands: bool = False) -> str:
    """'mean ± ci' with a fixed number of decimals; '—' for missing values."""
    if mean is None or not math.isfinite(mean):
        return "—"
    fmt = (lambda v: f"{v:,.0f}") if thousands else (lambda v: f"{v:.{digits}f}")
    if ci is None or not math.isfinite(ci):
        return fmt(mean)
    return f"{fmt(mean)} ± {fmt(ci)}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def provenance(results_dir: Path, exp: str) -> str:
    meta_path = results_dir / exp / "meta.json"
    if not meta_path.exists():
        return ""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    commit = (meta.get("git_commit") or "unknown")[:7]
    dirty = " (uncommitted changes)" if meta.get("git_dirty") else ""
    return (f"_Source: `{results_dir.name}/{exp}`, {meta.get('n_runs')} runs, "
            f"commit `{commit}`{dirty}, finished {meta.get('end_time')}._")


# --------------------------------------------------------------------------- sections


def section_diagnostics(results_dir: Path) -> str:
    summary, _ = load_results(results_dir, "e2_mesh_load")
    diag = mesh_diagnostics(summary)
    rows = [[mesh, int(d.topology_seed), int(d.n_bridges), f"{d.max_bridge_share:.3f}",
             "yes" if d.bottleneck_is_bridge else "no", f"{d.rho_sat:.3f}", f"{d.avg_degree:.2f}"]
            for mesh, d in diag.iterrows()]
    return "\n\n".join([
        "## E2 mesh diagnostics",
        "Topology-only diagnostics (no routing results). `max_bridge_share` is the largest "
        "fraction of ordered node pairs whose SPR path must cross a single bridge; `rho_sat` "
        "is the load at which SPR's busiest link saturates.",
        md_table(["Mesh", "Topology seed", "Bridges", "Max bridge share", "Busiest SPR link is a bridge",
                  "rho_sat", "Avg degree"], rows, "lrrrlrr"),
    ])


def section_e2(results_dir: Path) -> str:
    summary, _ = load_results(results_dir, "e2_mesh_load")
    metrics = ["outage_probability", "delivered_throughput_bps", "key_utilization"]
    stats = group_ci(summary, ["variant", "traffic.offered_load", "routing.strategy"], metrics)
    meshes = [m for m in MESH_ORDER if m in set(stats["variant"])]
    parts = ["## E2: SPR vs KAR vs offered load", provenance(results_dir, "e2_mesh_load")]
    for mesh in meshes:
        rows = []
        s = stats[stats["variant"] == mesh]
        for rho in sorted(s["traffic.offered_load"].unique()):
            by = s[s["traffic.offered_load"] == rho].set_index("routing.strategy")
            row = [f"{rho:g}"]
            for metric, digits, thousands in (("outage_probability", 3, False),
                                              ("delivered_throughput_bps", 0, True),
                                              ("key_utilization", 3, False)):
                for router in ("spr", "kar"):
                    r = by.loc[router]
                    row.append(pm(r[f"{metric}_mean"], r[f"{metric}_ci"], digits, thousands))
            rows.append(row)
        n = int(s["n"].max())
        parts += [f"### {mesh} (n = {n} seeds per point)",
                  md_table(["rho", "SPR outage", "KAR outage", "SPR throughput (bit/s)",
                            "KAR throughput (bit/s)", "SPR utilization", "KAR utilization"], rows)]
    return "\n\n".join(p for p in parts if p)


def section_e3(results_dir: Path, exp: str = "e3_mesh_density",
               heading: str = "## E3: mesh density") -> str:
    summary, _ = load_results(results_dir, exp)
    per = e3_per_degree(summary).sort_values("target_degree")
    rows = []
    for r in per.itertuples():
        reduction = "—" if not math.isfinite(r.relative_reduction) else f"{100 * r.relative_reduction:.1f}%"
        rows.append([f"{r.target_degree:g}", pm(r.achieved_degree, r.achieved_degree_ci, 2),
                     pm(r.spr_outage, r.spr_outage_ci), pm(r.kar_outage, r.kar_outage_ci), reduction,
                     pm(r.spr_throughput, r.spr_throughput_ci, thousands=True),
                     pm(r.kar_throughput, r.kar_throughput_ci, thousands=True),
                     f"{r.n_bridges:.1f}", f"{r.max_bridge_share:.3f}", f"{r.rho_sat:.3f}"])
    lam = summary["lambda_per_s"].round(6)
    if lam.nunique() == 1 and summary["offered_load"].round(6).nunique() > 1:
        load = (f"Fixed arrival rate lambda = {lam.iloc[0]:.4f} req/s (rho varies by topology: "
                f"{summary['offered_load'].min():.3f}–{summary['offered_load'].max():.3f})")
    else:
        load = f"rho = {summary['offered_load'].iloc[0]:g}"
    return "\n\n".join(p for p in [
        heading,
        provenance(results_dir, exp),
        f"{load}; {int(per['n_topologies'].max())} topologies per degree (topology seed = run "
        "seed), SPR and KAR paired on the same topologies. Relative reduction = 1 − mean KAR "
        "outage / mean SPR outage. Bridge columns are means over the topologies.",
        md_table(["Target degree", "Achieved degree", "SPR outage", "KAR outage", "Relative reduction",
                  "SPR throughput (bit/s)", "KAR throughput (bit/s)", "Mean bridges",
                  "Mean max bridge share", "Mean rho_sat"], rows),
    ] if p)


def section_e1(results_dir: Path) -> str:
    summary, _ = load_results(results_dir, "e1_chain_hops")
    metrics = ["outage_probability", "delivered_throughput_bps", "latency_mean_s", "latency_p95_s"]
    stats = group_ci(summary, ["traffic.offered_load", "routing.strategy"], metrics)
    rows = [[f"{r['traffic.offered_load']:g}", r["routing.strategy"].upper(), int(r["n"]),
             pm(r["outage_probability_mean"], r["outage_probability_ci"]),
             pm(r["delivered_throughput_bps_mean"], r["delivered_throughput_bps_ci"], thousands=True),
             pm(r["latency_mean_s_mean"], r["latency_mean_s_ci"], 4),
             pm(r["latency_p95_s_mean"], r["latency_p95_s_ci"], 4)]
            for _, r in stats.sort_values(["traffic.offered_load", "routing.strategy"],
                                          ascending=[True, False]).iterrows()]
    # SPR = KAR sanity check: compare each KAR run with the SPR run on the same rho and seed.
    checks = []
    for _, kar in summary[summary["routing.strategy"] == "kar"].iterrows():
        spr = summary[(summary["routing.strategy"] == "spr") & (summary["seed"] == kar["seed"])
                      & (summary["traffic.offered_load"] == kar["traffic.offered_load"])]
        cols = ["offered", "delivered", "outage"] + metrics
        same = (not spr.empty) and all(spr.iloc[0][c] == kar[c] or (pd.isna(spr.iloc[0][c]) and pd.isna(kar[c]))
                                       for c in cols)
        verdict = "identical to SPR" if same else ("no matching SPR run" if spr.empty else "DIFFERS from SPR")
        checks.append(f"rho = {kar['traffic.offered_load']:g}, seed {int(kar['seed'])}: {verdict}")
    sanity = (" KAR sanity run(s), no CI: " + "; ".join(checks) + ".") if checks else ""
    return "\n\n".join(p for p in [
        "## E1: chain (whole-run summary; per-hop breakdown is in the figures)",
        provenance(results_dir, "e1_chain_hops"),
        f"rho_sat = {summary['rho_sat'].iloc[0]:.3f}.{sanity}",
        md_table(["rho", "Router", "Seeds", "Outage", "Throughput (bit/s)", "Mean latency (s)",
                  "p95 latency (s)"], rows, "llrrrrr"),
    ] if p)


def section_e4(results_dir: Path) -> str:
    summary, _ = load_results(results_dir, "e4_secoqc")
    metrics = ["outage_probability", "delivered_throughput_bps", "key_utilization"]
    stats = group_ci(summary, ["traffic.offered_load", "routing.strategy"], metrics)
    rows = []
    for rho in sorted(stats["traffic.offered_load"].unique()):
        by = stats[stats["traffic.offered_load"] == rho].set_index("routing.strategy")
        rows.append([f"{rho:g}"] + [pm(by.loc[r]["outage_probability_mean"], by.loc[r]["outage_probability_ci"])
                                    for r in ("spr", "kar")]
                    + [pm(by.loc[r]["delivered_throughput_bps_mean"], by.loc[r]["delivered_throughput_bps_ci"],
                          thousands=True) for r in ("spr", "kar")])
    d = summary.iloc[0]
    return "\n\n".join(p for p in [
        "## E4: SECOQC (per-link measured rates)",
        provenance(results_dir, "e4_secoqc"),
        f"{int(d['n_bridges'])} bridges, max bridge share {d['max_bridge_share']:.3f}, "
        f"rho_sat = {d['rho_sat']:.3f}.",
        md_table(["rho", "SPR outage", "KAR outage", "SPR throughput (bit/s)", "KAR throughput (bit/s)"], rows),
    ] if p)


def section_e3b(results_dir: Path) -> str:
    return section_e3(results_dir, "e3b_mesh_density_fixed_lambda",
                      "## E3b: mesh density at a fixed arrival rate")


SECTIONS = [
    ("e2_mesh_load", section_diagnostics),
    ("e2_mesh_load", section_e2),
    ("e3_mesh_density", section_e3),
    ("e3b_mesh_density_fixed_lambda", section_e3b),
    ("e1_chain_hops", section_e1),
    ("e4_secoqc", section_e4),
]


def make_tables(results_dir: Path, out_path: Path) -> Path:
    parts = ["# Result tables",
             "Generated by `scripts/make_tables.py`. Values are mean ± 95% CI half-width across "
             "seeds, using Student's t (n − 1 degrees of freedom)."]
    for exp, section in SECTIONS:
        if not (results_dir / exp / "summary.csv").exists():
            parts.append(f"_{section.__name__.replace('section_', '').upper()}: no results in "
                         f"`{results_dir.name}/{exp}`; skipped._")
            continue
        parts.append(section(results_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(REPO_ROOT / "figures" / "tables.md"))
    args = parser.parse_args(argv)
    path = make_tables(Path(args.results_dir), Path(args.out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
