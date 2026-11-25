"""Aggregation helpers shared by scripts/make_plots.py and scripts/make_tables.py.

Confidence intervals are mean +/- t_{0.975, n-1} * s / sqrt(n) across seeds (Student's
t, since n = 5 seeds is small). Paired differences (KAR - SPR) are computed per seed
first, then summarized the same way.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Two-sided 95% critical values of Student's t, t_{0.975, df}, for df = 1..30.
_T975 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646,
    8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448,
    15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796,
    22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484,
    29: 2.0452, 30: 2.0423,
}
_Z975 = 1.959964

ROUTER_ORDER = ("spr", "kar")
ROUTER_LABEL = {"spr": "SPR", "kar": "KAR"}
MESH_ORDER = ("mesh0", "diverse")


def t_crit_95(df: int) -> float:
    """t_{0.975, df}. Exact table up to df=30, Cornish-Fisher expansion beyond (error < 1e-4)."""
    if df < 1:
        return float("nan")
    if df in _T975:
        return _T975[df]
    z = _Z975
    return (z + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2))


def mean_ci(values: Iterable[float]) -> tuple[float, float, int]:
    """(mean, 95% CI half-width, n) over the finite values; half-width NaN if n < 2."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(x[0]), float("nan"), 1
    half = t_crit_95(n - 1) * float(np.std(x, ddof=1)) / math.sqrt(n)
    return float(np.mean(x)), half, n


def group_ci(df: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """One row per group with <metric>_mean, <metric>_ci and n (rows per group)."""
    rows = []
    for key, group in df.groupby(list(keys), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_ci"], row[f"{m}_n"] = mean_ci(group[m])
        row["n"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(df: pd.DataFrame, keys: Sequence[str], metric: str,
                      a: str = "kar", b: str = "spr", pair_on: str = "seed",
                      router_col: str = "routing.strategy") -> pd.DataFrame:
    """Per group: mean and 95% CI of (metric[a] - metric[b]), paired on `pair_on`."""
    wide = df.pivot_table(index=list(keys) + [pair_on], columns=router_col, values=metric)
    diff = (wide[a] - wide[b]).rename("diff").reset_index()
    rows = []
    for key, group in diff.groupby(list(keys), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        mean, half, n = mean_ci(group["diff"])
        rows.append({**dict(zip(keys, key)), "diff_mean": mean, "diff_ci": half, "n": n})
    return pd.DataFrame(rows)


def load_results(results_dir: str | Path, exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, by_hops) for results/<exp>/. Raises FileNotFoundError with a hint."""
    exp_dir = Path(results_dir) / exp
    summary_path, hops_path = exp_dir / "summary.csv", exp_dir / "by_hops.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found; run: python experiments/run_experiment.py "
            f"experiments/configs/{exp}.yaml"
        )
    summary = pd.read_csv(summary_path)
    by_hops = pd.read_csv(hops_path) if hops_path.exists() else pd.DataFrame()
    return summary, by_hops


def key_size_bits(summary: pd.DataFrame) -> float:
    """K recovered from the summary: throughput = delivered * K / measured duration."""
    rows = summary[summary["delivered"] > 0]
    k = rows["delivered_throughput_bps"] * rows["measured_duration_s"] / rows["delivered"]
    k = k.round().unique()
    if len(k) != 1:
        raise ValueError(f"inconsistent key size across runs: {k}")
    return float(k[0])


def mesh_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per E2 mesh: topology seed, bridges, rho_sat, achieved degree."""
    cols = ["topology_seed", "n_bridges", "max_bridge_share", "bottleneck_is_bridge",
            "rho_sat", "avg_degree", "n_nodes", "n_links"]
    diag = summary.groupby("variant")[cols].first()
    order = [m for m in MESH_ORDER if m in diag.index] + [m for m in diag.index if m not in MESH_ORDER]
    return diag.loc[order]


def hop_bins(max_hops: int, n_bins: int = 6) -> list[tuple[int, int]]:
    """About n_bins contiguous hop ranges covering 1..max_hops, e.g. 31 -> 1-5, ..., 26-31.

    Bins have equal width round(max_hops / n_bins); a last bin narrower than half that
    width is merged into the previous one.
    """
    width = max(1, round(max_hops / n_bins))
    bins = [(lo, min(lo + width - 1, max_hops)) for lo in range(1, max_hops + 1, width)]
    if len(bins) > 1 and (bins[-1][1] - bins[-1][0] + 1) < width / 2:
        bins[-2:] = [(bins[-2][0], bins[-1][1])]
    return bins


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


def binned_by_hops(by_hops: pd.DataFrame, bins: Sequence[tuple[int, int]],
                   keys: Sequence[str]) -> pd.DataFrame:
    """Pool by_hops rows into hop bins, per `keys` + seed.

    outage_probability = outages / requests in the bin; latency_mean_s is weighted by the
    number of delivered requests per hop count (NaN when none were delivered).
    """
    df = by_hops.copy()
    edges = {h: i for i, (lo, hi) in enumerate(bins) for h in range(lo, hi + 1)}
    df["bin"] = df["hops"].map(edges)
    df = df.dropna(subset=["bin"])
    df["outages"] = df["count"] * df["outage_probability"]
    df["delivered"] = df["count"] - df["outages"]
    df["latency_x_delivered"] = df["latency_mean_s"].fillna(0.0) * df["delivered"]
    g = df.groupby(list(keys) + ["seed", "bin"], as_index=False)[
        ["count", "outages", "delivered", "latency_x_delivered"]].sum()
    g["outage_probability"] = g["outages"] / g["count"]
    g["latency_mean_s"] = np.where(g["delivered"] > 0,
                                   g["latency_x_delivered"] / g["delivered"].where(g["delivered"] > 0),
                                   np.nan)
    g["bin"] = g["bin"].astype(int)
    g["bin_label"] = g["bin"].map(lambda i: bin_label(*bins[i]))
    return g


def per_topology_reduction(summary: pd.DataFrame,
                           degree_key: str = "topology.target_avg_degree") -> pd.DataFrame:
    """One row per (degree, seed) topology: SPR/KAR outage, 1 - KAR/SPR, bridge share.

    Topologies where SPR has no outage are dropped (the reduction is undefined there).
    """
    wide = summary.pivot_table(index=[degree_key, "seed"], columns="routing.strategy",
                               values="outage_probability").reset_index()
    topo = (summary[summary["routing.strategy"] == "spr"]
            .set_index([degree_key, "seed"])[["max_bridge_share", "avg_degree", "n_bridges"]])
    wide = wide.join(topo, on=[degree_key, "seed"])
    wide = wide[wide["spr"] > 0].copy()
    wide["reduction"] = 1 - wide["kar"] / wide["spr"]
    return wide.rename(columns={degree_key: "target_degree"})


def relative_reduction(spr_mean: float, kar_mean: float) -> float:
    """1 - KAR/SPR on mean outage (NaN when SPR has no outage)."""
    return 1.0 - kar_mean / spr_mean if spr_mean > 0 else float("nan")


def e3_per_degree(summary: pd.DataFrame) -> pd.DataFrame:
    """Per target degree: achieved degree, SPR/KAR outage and throughput (mean, CI),
    relative outage reduction, paired KAR-SPR difference CI, mean bridge diagnostics."""
    key = "topology.target_avg_degree"
    stats = group_ci(summary, [key, "routing.strategy"],
                     ["outage_probability", "delivered_throughput_bps", "avg_degree"])
    diff = paired_difference(summary, [key], "outage_probability")
    rows = []
    for degree, g in stats.groupby(key):
        by = g.set_index("routing.strategy")
        topo = summary[(summary[key] == degree)]
        d = diff[diff[key] == degree].iloc[0]
        spr, kar = by.loc["spr"], by.loc["kar"]
        rows.append({
            "target_degree": degree,
            "achieved_degree": spr["avg_degree_mean"],
            "achieved_degree_ci": spr["avg_degree_ci"],
            "spr_outage": spr["outage_probability_mean"], "spr_outage_ci": spr["outage_probability_ci"],
            "kar_outage": kar["outage_probability_mean"], "kar_outage_ci": kar["outage_probability_ci"],
            "spr_throughput": spr["delivered_throughput_bps_mean"],
            "spr_throughput_ci": spr["delivered_throughput_bps_ci"],
            "kar_throughput": kar["delivered_throughput_bps_mean"],
            "kar_throughput_ci": kar["delivered_throughput_bps_ci"],
            "relative_reduction": relative_reduction(spr["outage_probability_mean"],
                                                     kar["outage_probability_mean"]),
            # CI of the reduction from the paired difference, with SPR's mean held fixed.
            "relative_reduction_ci": (d["diff_ci"] / spr["outage_probability_mean"]
                                      if spr["outage_probability_mean"] > 0 else float("nan")),
            "diff_mean": d["diff_mean"], "diff_ci": d["diff_ci"],
            "max_bridge_share": topo["max_bridge_share"].mean(),
            "n_bridges": topo["n_bridges"].mean(),
            "rho_sat": topo["rho_sat"].mean(),
            "n_topologies": int(spr["n"]),
        })
    return pd.DataFrame(rows)
