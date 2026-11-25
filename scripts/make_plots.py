"""Make the E1-E4 and topology figures from results/<exp>/summary.csv and by_hops.csv.

    python scripts/make_plots.py --all [--results-dir results] [--out-dir figures] [--ieee]
    python scripts/make_plots.py --exp e2_mesh_load

Every figure is written as PNG (200 dpi) and PDF. Error bars and bands are mean +/- 95%
CI across seeds using Student's t (qkdsim.analysis). SPR and KAR keep one fixed color and
marker in every figure. --ieee sizes figures for a 3.5 in IEEE column and appends
"_ieee" to the file names.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from qkdsim.analysis import (  # noqa: E402
    MESH_ORDER,
    ROUTER_LABEL,
    ROUTER_ORDER,
    bin_label,
    binned_by_hops,
    e3_per_degree,
    group_ci,
    hop_bins,
    key_size_bits,
    load_results,
    mesh_diagnostics,
    paired_difference,
    per_topology_reduction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Palette: categorical slots 1-2 of the reference data-viz palette (validated adjacent
# pair), plus chart ink. SPR and KAR never change color or marker.
ROUTER_COLOR = {"spr": "#257de0", "kar": "#f47230"}
ROUTER_MARKER = {"spr": "o", "kar": "s"}
INK, INK_2 = "#191919", "#666666"
GRID, AXIS = "#e7e7e7", "#c2c2c2"
NEUTRAL_SERIES = "#3d3d3d"  # derived single series (differences, reductions)
EDGE_GRAY, BRIDGE_RED = "#9c9c9c", "#ec4647"
RHO_RAMP = ["#84b9f3", "#2370cd", "#0e3d7e"]  # ordinal blue ramp (steps 250/500/700)
RHO_MARKERS = ["o", "s", "^", "D", "v"]

MESH_TITLE = {"mesh0": "mesh0 (default)", "diverse": "diverse"}
EXPERIMENTS = ("e1_chain_hops", "e2_mesh_load", "e3_mesh_density", "e3b_mesh_density_fixed_lambda",
               "e4_secoqc", "topology")
ALIASES = {"e1": "e1_chain_hops", "e2": "e2_mesh_load", "e3": "e3_mesh_density",
           "e3b": "e3b_mesh_density_fixed_lambda", "e4": "e4_secoqc", "topo": "topology"}


class Style:
    def __init__(self, ieee: bool) -> None:
        self.ieee = ieee
        self.suffix = "_ieee" if ieee else ""
        self.width = 3.5 if ieee else 7.0
        self.single_width = 3.5 if ieee else 5.0
        base = 7 if ieee else 9
        plt.rcParams.update({
            "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
            "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1, "figure.titlesize": base + 1,
            "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
            "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
            "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
            "lines.linewidth": 1.2 if ieee else 1.5, "lines.markersize": 3.5 if ieee else 5,
            "errorbar.capsize": 2, "legend.frameon": False, "savefig.dpi": 200,
            "font.family": "sans-serif",
        })

    def size(self, panels_wide: int, height_ratio: float) -> tuple[float, float]:
        width = self.width if panels_wide > 1 else self.single_width
        if self.ieee and panels_wide > 1:
            height_ratio *= 1.35  # narrow column: give side-by-side panels more height
        return width, width * height_ratio

    def wrap(self, text: str) -> str:
        """Wrap each line of a title to fit the IEEE column (unchanged otherwise)."""
        if not self.ieee:
            return text
        return "\n".join(textwrap.fill(line, 58) for line in text.split("\n"))

    @property
    def xlabel_rho(self) -> str:
        return "Offered load ρ" if self.ieee else "Offered load ρ (normalized by total key rate)"

    def label_size(self) -> float:
        return plt.rcParams["axes.labelsize"]


def save(fig: plt.Figure, out_dir: Path, name: str, style: Style) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}{style.suffix}.{ext}"
        # IEEE: keep the exact column width (a tight bbox can grow past 3.5 in); every figure
        # uses constrained layout, so content already fits inside the figure.
        fig.savefig(path, bbox_inches=None if style.ieee else "tight")
        paths.append(path)
    plt.close(fig)
    return paths


def router_errorbar(ax: plt.Axes, x, mean, ci, router: str, **kw) -> None:
    ax.errorbar(x, mean, yerr=np.nan_to_num(ci), color=ROUTER_COLOR[router],
                marker=ROUTER_MARKER[router], label=ROUTER_LABEL[router],
                markeredgecolor="white", markeredgewidth=0.6, **kw)


def router_legend(fig: plt.Figure, extra: Sequence[Line2D] = (), ncol: int | None = None) -> None:
    """Shared legend above the panels (figures use constrained layout)."""
    handles = [Line2D([], [], color=ROUTER_COLOR[r], marker=ROUTER_MARKER[r], label=ROUTER_LABEL[r],
                      markeredgecolor="white") for r in ROUTER_ORDER] + list(extra)
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol or len(handles))


# --------------------------------------------------------------------------- E1


def plot_e1(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    summary, hops = load_results(results_dir, "e1_chain_hops")
    hops = hops[hops["routing.strategy"] == "spr"]
    rhos = sorted(hops["traffic.offered_load"].unique())
    rho_sat = summary["rho_sat"].iloc[0]
    # Bin hop counts (~6 bins): per-hop curves are noisy at long paths, where few node
    # pairs exist (a 32-node chain has only 2 ordered pairs 31 hops apart).
    bins = hop_bins(int(hops["hops"].max()))
    binned = binned_by_hops(hops, bins, ["traffic.offered_load"])
    x = np.arange(len(bins))
    dodge = 0.08
    paths = []
    for metric, name, ylabel, note in (
        ("outage_probability", "e1_outage_vs_hops", "Outage probability", None),
        ("latency_mean_s", "e1_latency_vs_hops", "Mean latency (s)",
         "latency of delivered requests only"),
    ):
        fig, ax = plt.subplots(figsize=style.size(1, 0.62), layout="constrained")
        stats = group_ci(binned, ["traffic.offered_load", "bin"], [metric])
        for i, rho in enumerate(rhos):
            s = stats[stats["traffic.offered_load"] == rho].set_index("bin").reindex(range(len(bins)))
            color = RHO_RAMP[i % len(RHO_RAMP)]
            ax.errorbar(x + (i - (len(rhos) - 1) / 2) * dodge, s[f"{metric}_mean"],
                        yerr=np.nan_to_num(s[f"{metric}_ci"]), color=color,
                        marker=RHO_MARKERS[i % len(RHO_MARKERS)], markeredgecolor="white",
                        markeredgewidth=0.6, label=f"ρ = {rho:g}")
        ax.set_xticks(x, [bin_label(*b) for b in bins])
        ax.set_xlabel("SPR path length (hops, binned)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        title = f"Chain, {summary['n_nodes'].iloc[0]} nodes, SPR (KAR identical on a chain)"
        sub = f"ρ_sat = {rho_sat:.2f}; error bars: 95% CI across seeds"
        if note:
            sub = f"{note}; {sub}"
        # Lines below rho_sat coincide (no queueing); say so instead of hiding one silently.
        below = [f"{r:g}" for r in rhos if r < rho_sat]
        if len(below) > 1:
            sub += f"\nρ = {' and '.join(below)} lie on top of each other (below ρ_sat)"
        ax.set_title(style.wrap(f"{title}\n{sub}"), loc="left", color=INK)
        ax.title.set_fontsize(plt.rcParams["axes.titlesize"])
        ax.legend(loc="upper left", title="offered load")
        paths += save(fig, out_dir, name, style)
    return paths


# --------------------------------------------------------------------------- E2


def _e2_frames(results_dir: Path):
    summary, _ = load_results(results_dir, "e2_mesh_load")
    diag = mesh_diagnostics(summary)
    meshes = [m for m in MESH_ORDER if m in diag.index] or list(diag.index)
    return summary, diag, meshes


def _mesh_title(mesh: str, diag: pd.DataFrame, style: Style) -> str:
    d = diag.loc[mesh]
    if style.ieee:
        return (f"{mesh} (seed {int(d['topology_seed'])})\n"
                f"{int(d['n_bridges'])} bridges, max share {d['max_bridge_share']:.2f}")
    return (f"{MESH_TITLE.get(mesh, mesh)}, topology seed {int(d['topology_seed'])}\n"
            f"{int(d['n_bridges'])} bridges, max bridge share {d['max_bridge_share']:.2f}")


def _rho_sat_line(ax: plt.Axes, rho_sat: float, label: str | None = "top") -> None:
    """Dashed vertical line at rho_sat, labeled at the top or bottom of the axes."""
    ax.axvline(rho_sat, color=INK_2, linestyle="--", linewidth=0.9)
    if label is None:
        return
    top = label == "top"
    ax.annotate(f"ρ_sat = {rho_sat:.3f}", xy=(rho_sat, 1.0 if top else 0.0),
                xycoords=("data", "axes fraction"), xytext=(3, -3 if top else 3),
                textcoords="offset points", ha="left", va="top" if top else "bottom",
                color=INK_2, fontsize=plt.rcParams["legend.fontsize"])


def _e2_panels(results_dir: Path, out_dir: Path, style: Style, metric: str, ylabel: str,
               name: str, decorate: Callable[[plt.Axes, pd.DataFrame, str], list[Line2D]] | None = None,
               ylim0: bool = True) -> list[Path]:
    summary, diag, meshes = _e2_frames(results_dir)
    stats = group_ci(summary, ["variant", "traffic.offered_load", "routing.strategy"], [metric])
    fig, axes = plt.subplots(1, len(meshes), figsize=style.size(2, 0.42), sharey=True,
                             squeeze=False, layout="constrained")
    extra: list[Line2D] = []
    for ax, mesh in zip(axes[0], meshes):
        for router in ROUTER_ORDER:
            s = stats[(stats["variant"] == mesh) & (stats["routing.strategy"] == router)]
            s = s.sort_values("traffic.offered_load")
            router_errorbar(ax, s["traffic.offered_load"], s[f"{metric}_mean"], s[f"{metric}_ci"], router)
        _rho_sat_line(ax, diag.loc[mesh, "rho_sat"])
        if decorate:
            extra = decorate(ax, summary[summary["variant"] == mesh], mesh) or extra
        ax.set_title(_mesh_title(mesh, diag, style), loc="left")
    if ylim0:  # after all panels are drawn, so the shared limits cover every panel
        axes[0][0].set_ylim(bottom=0)
    axes[0][0].set_ylabel(ylabel)
    fig.supxlabel(style.xlabel_rho, fontsize=style.label_size())
    router_legend(fig, extra)
    return save(fig, out_dir, name, style)


def plot_e2(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    paths = _e2_panels(results_dir, out_dir, style, "outage_probability", "Outage probability",
                       "e2_outage_vs_rho")

    summary_all, _, _ = _e2_frames(results_dir)
    k_bits = key_size_bits(summary_all)

    def offered_line(ax: plt.Axes, s: pd.DataFrame, mesh: str) -> list[Line2D]:
        off = s[s["routing.strategy"] == "spr"].groupby("traffic.offered_load")["lambda_per_s"].mean()
        ax.plot(off.index, off.to_numpy() * k_bits, color=INK, linestyle=":", linewidth=1.0, zorder=1)
        return [Line2D([], [], color=INK, linestyle=":", label="offered load (λK)")]

    paths += _e2_panels(results_dir, out_dir, style, "delivered_throughput_bps",
                        "Delivered key throughput (bit/s)", "e2_throughput_vs_rho", offered_line)
    paths += _e2_panels(results_dir, out_dir, style, "key_utilization",
                        "Network key utilization\n(consumed / generated)", "e2_utilization_vs_rho")
    paths += plot_e2_latency(results_dir, out_dir, style)
    paths += plot_e2_paired_gain(results_dir, out_dir, style)
    return paths


def plot_e2_latency(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    """p95 latency (top) above outage (bottom), shared x per mesh (Addendum C)."""
    summary, diag, meshes = _e2_frames(results_dir)
    metrics = ["latency_p95_s", "outage_probability"]
    stats = group_ci(summary, ["variant", "traffic.offered_load", "routing.strategy"], metrics)
    timeout = None
    fig, axes = plt.subplots(2, len(meshes), figsize=style.size(2, 0.72), sharex="col",
                             sharey="row", squeeze=False, layout="constrained")
    for col, mesh in enumerate(meshes):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            for router in ROUTER_ORDER:
                s = stats[(stats["variant"] == mesh) & (stats["routing.strategy"] == router)]
                s = s.sort_values("traffic.offered_load")
                router_errorbar(ax, s["traffic.offered_load"], s[f"{metric}_mean"], s[f"{metric}_ci"], router)
            # Label rho_sat once per column, in the outage panel (p95 sits at the top).
            _rho_sat_line(ax, diag.loc[mesh, "rho_sat"], label="top" if row == 1 else None)
        axes[0][col].set_title(_mesh_title(mesh, diag, style), loc="left")
        axes[0][col].set_yscale("log")
    axes[1][0].set_ylim(bottom=0)
    axes[0][0].set_ylabel("p95 latency, delivered\nrequests (s, log scale)")
    axes[1][0].set_ylabel("Outage probability")
    fig.supxlabel(style.xlabel_rho, fontsize=style.label_size())
    router_legend(fig)
    return save(fig, out_dir, "e2_latency_vs_rho", style)


def plot_e2_paired_gain(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    summary, diag, meshes = _e2_frames(results_dir)
    diff = paired_difference(summary, ["variant", "traffic.offered_load"], "outage_probability")
    fig, axes = plt.subplots(1, len(meshes), figsize=style.size(2, 0.42), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, mesh in zip(axes[0], meshes):
        d = diff[diff["variant"] == mesh].sort_values("traffic.offered_load")
        ax.axhline(0.0, color=INK_2, linewidth=0.9)
        ax.errorbar(d["traffic.offered_load"], d["diff_mean"], yerr=np.nan_to_num(d["diff_ci"]),
                    color=NEUTRAL_SERIES, marker="o", markeredgecolor="white", markeredgewidth=0.6)
        _rho_sat_line(ax, diag.loc[mesh, "rho_sat"], label=None)  # value given in the title
        ax.set_title(f"{_mesh_title(mesh, diag, style)}\nρ_sat = {diag.loc[mesh, 'rho_sat']:.3f} (dashed)",
                     loc="left")
    axes[0][0].set_ylabel("KAR − SPR outage\n(paired by seed; < 0: KAR better)")
    fig.supxlabel(style.xlabel_rho, fontsize=style.label_size())
    n = int(diff["n"].max())
    fig.suptitle(f"Paired difference, mean ± 95% CI (t, n = {n} seeds)",
                 x=0.01, ha="left", color=INK_2, fontsize=plt.rcParams["legend.fontsize"])
    return save(fig, out_dir, "e2_paired_gain", style)


# --------------------------------------------------------------------------- E3


E3_EXP, E3B_EXP = "e3_mesh_density", "e3b_mesh_density_fixed_lambda"
DEGREE_RAMP = ["#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]  # ordinal blue ramp, steps 250-700


def load_text(summary: pd.DataFrame) -> str:
    """'ρ = 0.3' for fixed-rho sweeps, 'λ = 95.2 req/s' when the arrival rate is fixed."""
    lam = summary["lambda_per_s"].round(6)
    if lam.nunique() == 1 and summary["offered_load"].round(6).nunique() > 1:
        return f"λ = {lam.iloc[0]:.1f} req/s (fixed)"
    return f"ρ = {summary['offered_load'].iloc[0]:g}"


def plot_e3(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    paths = plot_e3_family(results_dir, out_dir, style, E3_EXP, "e3")
    return paths + plot_gain_vs_bridge_share(results_dir, out_dir, style)


def plot_e3b(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    return plot_e3_family(results_dir, out_dir, style, E3B_EXP, "e3b")


def plot_e3_family(results_dir: Path, out_dir: Path, style: Style, exp: str,
                   prefix: str) -> list[Path]:
    """Outage, throughput and KAR gain vs achieved degree, for E3 or E3b."""
    summary, _ = load_results(results_dir, exp)
    per = e3_per_degree(summary).sort_values("achieved_degree")
    load = load_text(summary)
    n_topo = int(per["n_topologies"].max())
    dodge = 0.05 * (per["achieved_degree"].max() - per["achieved_degree"].min()) / 2
    paths = []
    for metric, name, ylabel in (
        ("outage", f"{prefix}_outage_vs_degree", "Outage probability"),
        ("throughput", f"{prefix}_throughput_vs_degree", "Delivered key throughput (bit/s)"),
    ):
        fig, ax = plt.subplots(figsize=style.size(1, 0.62), layout="constrained")
        for i, router in enumerate(ROUTER_ORDER):  # small x offset so the CIs don't overlap
            router_errorbar(ax, per["achieved_degree"] + (i - 0.5) * dodge, per[f"{router}_{metric}"],
                            per[f"{router}_{metric}_ci"], router)
        ax.set_xlabel("Achieved mean node degree")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(style.wrap(f"Random mesh, n = {summary['n_nodes'].iloc[0]}, {load}; "
                                f"{n_topo} topologies per degree, 95% CI"), loc="left")
        ax.legend(loc="best")
        paths += save(fig, out_dir, name, style)

    # Relative reduction: ratio of means (line) plus each topology's own 1 - KAR/SPR (dots),
    # shown instead of a CI because per-topology reductions are bounded and very spread out.
    wide = per_topology_reduction(summary)
    degree_x = dict(zip(per["target_degree"], per["achieved_degree"]))
    fig, ax = plt.subplots(figsize=style.size(1, 0.62), layout="constrained")
    ax.axhline(0.0, color=INK_2, linewidth=0.9)
    # Fixed sideways offsets (not random) so stacked values, e.g. several 100%, stay countable;
    # each degree's dots stay within half the smallest gap between degrees.
    key = "target_degree"
    sizes = wide.groupby(key)[key].transform("size")
    rank = wide.groupby(key).cumcount() - (sizes - 1) / 2
    xs = np.sort(per["achieved_degree"].to_numpy())
    gap = np.diff(xs).min() if len(xs) > 1 else 1.0
    spread = 0.5 * gap / max(int(sizes.max()) if len(sizes) else 1, 2)
    ax.scatter(wide[key].map(degree_x) + rank * spread, 100 * wide["reduction"], s=10 if n_topo > 8 else 14,
               color=EDGE_GRAY, edgecolors="white", linewidths=0.5, zorder=2, label="one topology")
    ax.plot(per["achieved_degree"], 100 * per["relative_reduction"], color=NEUTRAL_SERIES,
            marker="o", markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            label="all topologies (1 − mean KAR / mean SPR)")
    ax.set_xticks(per["achieved_degree"],
                  [f"{r.achieved_degree:.2f}\n({r.max_bridge_share:.2f})" for r in per.itertuples()])
    ax.set_xlabel("Achieved mean node degree\n(in parentheses: mean max bridge share)")
    ax.set_ylabel("KAR outage reduction vs SPR (%)")
    ax.set_ylim(min(-5, ax.get_ylim()[0]), 105)
    n_dropped = n_topo * len(per) - len(wide)
    dropped = f"; {n_dropped} topologies with no SPR outage omitted" if n_dropped else ""
    ax.set_title(style.wrap(f"Random mesh, n = {summary['n_nodes'].iloc[0]}, {load}; "
                            f"{n_topo} topologies per degree{dropped}"), loc="left")
    ax.legend(loc="upper left")
    paths += save(fig, out_dir, f"{prefix}_kar_gain_vs_degree", style)
    return paths


def plot_gain_vs_bridge_share(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    """Headline figure: KAR's relative outage reduction vs max bridge share, one dot per
    topology, colored by target degree. Left: E3 (fixed rho); right: E3b (fixed lambda)."""
    panels = [(E3_EXP, "E3"), (E3B_EXP, "E3b")]
    fig, axes = plt.subplots(1, 2, figsize=style.size(2, 0.45), sharey=True, sharex=True,
                             squeeze=False, layout="constrained")
    degrees: list[float] = []
    for ax, (exp, label) in zip(axes[0], panels):
        ax.axhline(0.0, color=INK_2, linewidth=0.9)
        if not (results_dir / exp / "summary.csv").exists():
            ax.text(0.5, 0.5, f"no {label} results yet", transform=ax.transAxes,
                    ha="center", va="center", color=INK_2)
            ax.set_title(label, loc="left")
            continue
        summary, _ = load_results(results_dir, exp)
        topo = per_topology_reduction(summary)
        degrees = sorted(set(degrees) | set(topo["target_degree"]))
        for i, degree in enumerate(sorted(topo["target_degree"].unique())):
            t = topo[topo["target_degree"] == degree]
            ax.scatter(t["max_bridge_share"], 100 * t["reduction"],
                       color=DEGREE_RAMP[i % len(DEGREE_RAMP)], marker=RHO_MARKERS[i % len(RHO_MARKERS)],
                       s=14 if style.ieee else 22, edgecolors="white", linewidths=0.5, zorder=2,
                       alpha=0.85)  # topologies can share a bridge share; keep overlaps visible
        n_total = summary[summary["routing.strategy"] == "spr"].shape[0]
        omitted = n_total - len(topo)
        note = f"; {omitted} with no SPR outage omitted" if omitted else ""
        ax.set_title(style.wrap(f"{label}: {load_text(summary)}\n{len(topo)} topologies{note}"), loc="left")
    axes[0][0].set_ylabel("KAR outage reduction vs SPR (%)")
    fig.supxlabel("Max bridge share (fraction of node pairs whose SPR path crosses the largest bridge)"
                  if not style.ieee else "Max bridge share", fontsize=style.label_size())
    handles = [Line2D([], [], color=DEGREE_RAMP[i % len(DEGREE_RAMP)], marker=RHO_MARKERS[i % len(RHO_MARKERS)],
                      linestyle="", markeredgecolor="white", label=f"{d:g}") for i, d in enumerate(degrees)]
    if handles:
        fig.legend(handles=handles, title="target degree", loc="outside upper center", ncol=len(handles))
    return save(fig, out_dir, "e3_gain_vs_bridge_share", style)


# --------------------------------------------------------------------------- E4


def plot_e4(results_dir: Path, out_dir: Path, style: Style) -> list[Path]:
    summary, _ = load_results(results_dir, "e4_secoqc")
    metrics = [("outage_probability", "Outage probability"),
               ("delivered_throughput_bps",
                "Throughput (bit/s)" if style.ieee else "Delivered key throughput (bit/s)")]
    stats = group_ci(summary, ["traffic.offered_load", "routing.strategy"], [m for m, _ in metrics])
    rhos = sorted(stats["traffic.offered_load"].unique())
    x = np.arange(len(rhos))
    width = 0.38  # two bars per group leave a gap between them
    fig, axes = plt.subplots(1, 2, figsize=style.size(2, 0.42), squeeze=False, layout="constrained")
    for ax, (metric, ylabel) in zip(axes[0], metrics):
        for i, router in enumerate(ROUTER_ORDER):
            s = stats[stats["routing.strategy"] == router].set_index("traffic.offered_load").loc[rhos]
            ax.bar(x + (i - 0.5) * (width + 0.02), s[f"{metric}_mean"], width,
                   yerr=np.nan_to_num(s[f"{metric}_ci"]), color=ROUTER_COLOR[router],
                   label=ROUTER_LABEL[router], error_kw={"ecolor": INK, "elinewidth": 0.8})
        ax.set_xticks(x, [f"{r:g}" for r in rhos])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    fig.supxlabel(style.xlabel_rho, fontsize=style.label_size())
    d = summary.iloc[0]
    diag = (f"{int(d['n_bridges'])} bridges, max bridge share {d['max_bridge_share']:.2f}, "
            f"ρ_sat = {d['rho_sat']:.2f}")
    head = "SECOQC Vienna" if style.ieee else "SECOQC Vienna (6 nodes, 8 links, measured per-link rates)"
    fig.suptitle(f"{head}:\n{diag}" if style.ieee else f"{head}: {diag}", x=0.01, ha="left",
                 fontsize=plt.rcParams["axes.titlesize"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROUTER_COLOR[r], label=ROUTER_LABEL[r]) for r in ROUTER_ORDER]
    axes[0][1].legend(handles=handles, loc="upper left")
    return save(fig, out_dir, "e4_secoqc_bars", style)


# --------------------------------------------------------------------------- topology


def plot_topology(results_dir: Path, out_dir: Path, style: Style,
                  base_config: Path = REPO_ROOT / "configs" / "default.yaml") -> list[Path]:
    """Both E2 meshes, rebuilt from configs/default.yaml with each mesh's topology seed."""
    from qkdsim.config import load_config
    from qkdsim.simulator import build_network
    from qkdsim.traffic import resolve_arrival_rate, run_rng_streams, spr_edge_shares, traffic_pairs
    import networkx as nx

    _, diag, meshes = _e2_frames(results_dir)
    graphs = []
    for mesh in meshes:
        seed = int(diag.loc[mesh, "topology_seed"])
        cfg = load_config(base_config, overrides={"topology": {"seed": seed}})
        G = build_network(cfg, run_rng_streams(0, seed)[0])
        shares = spr_edge_shares(G, traffic_pairs(G, cfg.traffic))
        load = resolve_arrival_rate(G, cfg.traffic)
        if (load.n_bridges, round(load.max_bridge_share, 6)) != (
                int(diag.loc[mesh, "n_bridges"]), round(float(diag.loc[mesh, "max_bridge_share"]), 6)):
            print(f"WARNING: rebuilt {mesh} differs from the results' diagnostics "
                  f"(was configs/default.yaml changed?)", file=sys.stderr)
        graphs.append((mesh, G, shares, {frozenset(e) for e in nx.bridges(G)}))

    max_share = max(max(s.values()) for _, _, s, _ in graphs)
    fig, axes = plt.subplots(1, len(graphs), figsize=style.size(2, 0.55), squeeze=False,
                             layout="constrained")
    for ax, (mesh, G, shares, bridges) in zip(axes[0], graphs):
        pos = nx.get_node_attributes(G, "pos")
        for is_bridge in (False, True):  # bridges drawn on top
            edges = [(u, v) for u, v in G.edges if (frozenset((u, v)) in bridges) == is_bridge]
            segs = [(pos[u], pos[v]) for u, v in edges]
            widths = [0.4 + (5.0 if not style.ieee else 3.0) * shares.get((u, v), 0.0) / max_share
                      for u, v in edges]
            ax.add_collection(LineCollection(segs, linewidths=widths, capstyle="round",
                                             colors=BRIDGE_RED if is_bridge else EDGE_GRAY,
                                             zorder=2 if is_bridge else 1))
        xy = np.array([pos[n] for n in G.nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=10 if style.ieee else 16, color=INK, zorder=3,
                   edgecolors="white", linewidths=0.6)
        busiest = max(shares, key=shares.get)
        (x1, y1), (x2, y2) = pos[busiest[0]], pos[busiest[1]]
        ax.annotate(f"busiest: b = {shares[busiest]:.2f}", xy=((x1 + x2) / 2, (y1 + y2) / 2),
                    xytext=(6, 6), textcoords="offset points", fontsize=plt.rcParams["legend.fontsize"],
                    color=INK, bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": AXIS, "lw": 0.5},
                    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.6}, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 52)
        ax.set_xlabel("x (km)")
        ax.set_title(_mesh_title(mesh, diag, style), loc="left")
    axes[0][0].set_ylabel("y (km)")
    handles = [
        Line2D([], [], color=EDGE_GRAY, linewidth=3, label="link (width ∝ SPR pair share b_e)"),
        Line2D([], [], color=BRIDGE_RED, linewidth=3, label="bridge (no alternative path)"),
        Line2D([], [], color=INK, marker="o", linestyle="", markeredgecolor="white", label="trusted node"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3 if not style.ieee else 1)
    return save(fig, out_dir, "topo_mesh0_vs_diverse", style)


# --------------------------------------------------------------------------- CLI

PLOTTERS: dict[str, Callable[[Path, Path, Style], list[Path]]] = {
    "e1_chain_hops": plot_e1,
    "e2_mesh_load": plot_e2,
    "e3_mesh_density": plot_e3,  # also e3_gain_vs_bridge_share (E3b panel if available)
    "e3b_mesh_density_fixed_lambda": plot_e3b,
    "e4_secoqc": plot_e4,
    "topology": plot_topology,
}
# Which results directory each plotter reads (for --all skipping).
SOURCE = {"e1_chain_hops": "e1_chain_hops", "e2_mesh_load": "e2_mesh_load",
          "e3_mesh_density": "e3_mesh_density",
          "e3b_mesh_density_fixed_lambda": "e3b_mesh_density_fixed_lambda",
          "e4_secoqc": "e4_secoqc", "topology": "e2_mesh_load"}


def make_plots(names: Sequence[str], results_dir: Path, out_dir: Path, ieee: bool = False,
               skip_missing: bool = False) -> list[Path]:
    style = Style(ieee)
    written: list[Path] = []
    for name in names:
        if skip_missing and not (results_dir / SOURCE[name] / "summary.csv").exists():
            print(f"skipping {name}: no {results_dir / SOURCE[name] / 'summary.csv'}")
            continue
        paths = PLOTTERS[name](results_dir, out_dir, style)
        written += paths
        for p in paths:
            if p.suffix == ".png":
                print(f"wrote {p.with_suffix('')}.{{png,pdf}}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--exp", help=f"one of {list(EXPERIMENTS)} (or e1..e4, topo)")
    which.add_argument("--all", action="store_true", help="every experiment with results")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--ieee", action="store_true", help="3.5 in single-column width")
    args = parser.parse_args(argv)

    if args.all:
        names, skip = list(EXPERIMENTS), True
    else:
        name = ALIASES.get(args.exp, args.exp)
        if name not in PLOTTERS:
            parser.error(f"unknown experiment {args.exp!r}; choose from {list(EXPERIMENTS)}")
        names, skip = [name], False
    make_plots(names, Path(args.results_dir), Path(args.out_dir), args.ieee, skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
