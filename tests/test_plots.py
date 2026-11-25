"""Tests for the analysis helpers, scripts/make_plots.py and scripts/make_tables.py (M5).

All inputs are synthetic results written to a pytest temporary directory; nothing here
reads results/. The column layout matches the runner's summary.csv / by_hops.csv because
the template row comes from metrics.summarize on a tiny real simulation.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qkdsim.analysis import (
    bin_label,
    binned_by_hops,
    e3_per_degree,
    hop_bins,
    key_size_bits,
    mean_ci,
    paired_difference,
    per_topology_reduction,
    t_crit_95,
)
from qkdsim.config import config_from_dict
from qkdsim.metrics import by_hops, summarize
from qkdsim.simulator import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
K_BITS = 256
WINDOW_S = 540.0

FIGURES = {
    "e1_chain_hops": ["e1_outage_vs_hops", "e1_latency_vs_hops"],
    "e2_mesh_load": ["e2_outage_vs_rho", "e2_throughput_vs_rho", "e2_latency_vs_rho",
                     "e2_utilization_vs_rho", "e2_paired_gain"],
    "e3_mesh_density": ["e3_outage_vs_degree", "e3_throughput_vs_degree", "e3_kar_gain_vs_degree",
                        "e3_gain_vs_bridge_share"],
    "e3b_mesh_density_fixed_lambda": ["e3b_outage_vs_degree", "e3b_throughput_vs_degree",
                                      "e3b_kar_gain_vs_degree"],
    "e4_secoqc": ["e4_secoqc_bars"],
    "topology": ["topo_mesh0_vs_diverse"],
}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- synthetic data


@pytest.fixture(scope="module")
def template():
    """(summary row, by_hops columns) from a real tiny run: the schema the runner writes."""
    cfg = config_from_dict({"topology": {"type": "chain", "n_nodes": 4},
                            "simulation": {"duration_s": 20, "warmup_s": 5}})
    result = run_simulation(cfg, 0)
    return summarize(result, cfg), list(by_hops(result, cfg).columns)


def fake_run(template_row: dict, params: dict, seed: int, rng: np.random.Generator,
             outage: float, lam: float, **fields) -> dict:
    offered = int(lam * WINDOW_S)
    delivered = int(round(offered * (1 - outage)))
    row = dict(template_row)
    row.update({
        "seed": seed, "strategy": params["routing.strategy"], "measured_duration_s": WINDOW_S,
        "lambda_per_s": lam, "offered": offered, "delivered": delivered,
        "outage": offered - delivered, "no_path": 0, "outage_probability": (offered - delivered) / offered,
        "delivered_throughput_bps": delivered * K_BITS / WINDOW_S,
        "latency_mean_s": 0.01 + 2 * outage, "latency_p95_s": 0.02 + 4 * outage,
        "key_utilization": float(rng.uniform(0.05, 0.6)),
    })
    row.update(fields)
    return row


def write_experiment(root: Path, exp: str, runs: list[tuple[dict, dict]], hop_cols: list[str],
                     variants: bool = False) -> None:
    """runs: (params, summary row). Writes summary.csv and by_hops.csv like the runner."""
    exp_dir = root / exp
    exp_dir.mkdir(parents=True)
    rows, hop_rows = [], []
    for i, (params, row) in enumerate(runs):
        ident = {"run_id": f"run{i}", "variant": params.pop("variant", None), **params, "seed": row["seed"]}
        rows.append({**ident, "wall_s": 1.0, **row})
        for h in range(1, 5):
            hop_rows.append({**ident, **dict(zip(hop_cols, [h, 10 * h, 0.02 * h * (row["outage_probability"] > 0),
                                                         0.01 * h]))})
    pd.DataFrame(rows).to_csv(exp_dir / "summary.csv", index=False)
    pd.DataFrame(hop_rows).to_csv(exp_dir / "by_hops.csv", index=False)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory, template) -> Path:
    row0, hop_cols = template
    root = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    seeds = [0, 1, 2]

    e1 = [({"traffic.offered_load": rho, "routing.strategy": "spr"},
           fake_run(row0, {"routing.strategy": "spr"}, s, rng, max(0, rho - 0.6) * 0.3 + 0.001 * s, 20 * rho,
                    rho_sat=0.69, n_nodes=32))
          for rho in (0.2, 0.5, 0.8) for s in seeds]
    e1.append(({"traffic.offered_load": 0.5, "routing.strategy": "kar"},
               fake_run(row0, {"routing.strategy": "kar"}, 0, rng, 0.0, 10, rho_sat=0.69, n_nodes=32)))
    write_experiment(root, "e1_chain_hops", e1, hop_cols)

    meshes = {"mesh0": dict(topology_seed=0, n_bridges=4, max_bridge_share=0.434483, rho_sat=0.156,
                            bottleneck_is_bridge=True, avg_degree=4.07, n_nodes=30, n_links=61),
              "diverse": dict(topology_seed=8, n_bridges=5, max_bridge_share=0.128736, rho_sat=0.172,
                              bottleneck_is_bridge=False, avg_degree=4.13, n_nodes=30, n_links=62)}
    e2 = []
    for mesh, diag in meshes.items():
        for rho in (0.1, 0.3, 0.6):
            for router in ("spr", "kar"):
                for s in seeds:
                    base = max(0.0, rho - diag["rho_sat"]) * (0.8 if router == "kar" and mesh == "diverse" else 1)
                    e2.append(({"variant": mesh, "topology.seed": diag["topology_seed"],
                                "traffic.offered_load": rho, "routing.strategy": router},
                               fake_run(row0, {"routing.strategy": router}, s, rng, base + 0.002 * s,
                                        100 * rho, offered_load=rho, **diag)))
    write_experiment(root, "e2_mesh_load", e2, hop_cols)

    for exp, fixed_lambda in (("e3_mesh_density", False), ("e3b_mesh_density_fixed_lambda", True)):
        e3 = []
        for degree, mbs in ((3, 0.47), (4, 0.36), (6, 0.11)):
            for router in ("spr", "kar"):
                for s in seeds:
                    out = 0.2 - (0.02 * degree if router == "kar" else 0) + 0.01 * s
                    # E3b: lambda fixed, so the derived rho differs per topology.
                    lam, rho = (95.1826, 0.3 * 4 / degree + 0.01 * s) if fixed_lambda else (10.0 * degree, 0.3)
                    e3.append(({"topology.target_avg_degree": degree, "routing.strategy": router},
                               fake_run(row0, {"routing.strategy": router}, s, rng, out, lam, offered_load=rho,
                                        avg_degree=degree + 0.1 * s, max_bridge_share=mbs + 0.01 * s,
                                        n_bridges=5 - degree // 2, rho_sat=0.18, n_nodes=30)))
        write_experiment(root, exp, e3, hop_cols)

    e4 = [({"traffic.offered_load": rho, "routing.strategy": router},
           fake_run(row0, {"routing.strategy": router}, s, rng, 0.2 + rho / 4 - (0.03 if router == "kar" else 0),
                    10 * rho, n_bridges=2, max_bridge_share=0.333, rho_sat=0.10, offered_load=rho))
          for rho in (0.3, 0.6, 0.9) for router in ("spr", "kar") for s in seeds]
    write_experiment(root, "e4_secoqc", e4, hop_cols)
    return root


def test_synthetic_matches_runner_schema(results_dir: Path, template) -> None:
    row0, hop_cols = template
    summary = pd.read_csv(results_dir / "e2_mesh_load" / "summary.csv")
    assert set(row0) <= set(summary.columns)
    assert {"run_id", "variant", "seed", "wall_s", "routing.strategy", "traffic.offered_load"} <= set(summary.columns)
    hops = pd.read_csv(results_dir / "e1_chain_hops" / "by_hops.csv")
    assert set(hop_cols) <= set(hops.columns)


# --------------------------------------------------------------------------- analysis


def test_t_critical_values() -> None:
    assert t_crit_95(4) == pytest.approx(2.7764)  # n = 5 seeds
    assert t_crit_95(2) == pytest.approx(4.3027)
    assert abs(t_crit_95(31) - 2.0395) < 1e-3  # expansion continues the table smoothly
    assert t_crit_95(1000) == pytest.approx(1.9623, abs=1e-3)
    assert t_crit_95(4) > 1.96


def test_mean_ci() -> None:
    mean, half, n = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (mean, n) == (3.0, 5)
    assert half == pytest.approx(2.7764 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
    assert np.isnan(mean_ci([2.0])[1])
    assert mean_ci([np.nan, 1.0, 3.0])[2] == 2  # NaNs ignored


def test_paired_difference() -> None:
    df = pd.DataFrame({"g": [1] * 6, "seed": [0, 1, 2] * 2,
                       "routing.strategy": ["spr"] * 3 + ["kar"] * 3,
                       "y": [0.5, 0.6, 0.7, 0.4, 0.5, 0.6]})
    d = paired_difference(df, ["g"], "y").iloc[0]
    assert d["diff_mean"] == pytest.approx(-0.1)
    assert d["diff_ci"] == pytest.approx(0.0, abs=1e-12)  # identical per-seed differences


def test_hop_bins() -> None:
    bins = hop_bins(31)  # 32-node chain
    assert bins == [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 31)]
    assert [bin_label(*b) for b in bins][-1] == "26–31"
    assert hop_bins(4) == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert hop_bins(33) == [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 33)]
    for n in range(1, 60):  # bins always tile 1..n exactly
        b = hop_bins(n)
        assert b[0][0] == 1 and b[-1][1] == n
        assert all(hi + 1 == lo for (_, hi), (lo, _) in zip(b, b[1:]))


def test_binned_by_hops_pools_requests() -> None:
    df = pd.DataFrame({"rho": [0.8] * 3, "seed": [0] * 3, "hops": [1, 2, 3],
                       "count": [100, 50, 10], "outage_probability": [0.0, 0.2, 1.0],
                       "latency_mean_s": [1.0, 2.0, np.nan]})
    g = binned_by_hops(df, [(1, 2), (3, 3)], ["rho"]).set_index("bin")
    assert g.loc[0, "outage_probability"] == pytest.approx(10 / 150)  # pooled, not averaged
    assert g.loc[0, "latency_mean_s"] == pytest.approx((100 * 1.0 + 40 * 2.0) / 140)  # per delivered
    assert np.isnan(g.loc[1, "latency_mean_s"])  # nothing delivered at 3 hops
    assert list(g["bin_label"]) == ["1–2", "3"]


def test_per_topology_reduction(results_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "e3_mesh_density" / "summary.csv")
    topo = per_topology_reduction(summary)
    assert len(topo) == 3 * 3  # 3 degrees x 3 topology seeds, SPR outage > 0 in all
    row = topo.iloc[0]
    assert row["reduction"] == pytest.approx(1 - row["kar"] / row["spr"])
    assert {"max_bridge_share", "target_degree", "seed"} <= set(topo.columns)


def test_e3_per_degree_and_key_size(results_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "e3_mesh_density" / "summary.csv")
    per = e3_per_degree(summary)
    assert list(per["target_degree"]) == [3, 4, 6]
    row = per[per["target_degree"] == 6].iloc[0]
    assert row["relative_reduction"] == pytest.approx(1 - row["kar_outage"] / row["spr_outage"])
    assert key_size_bits(summary) == K_BITS


# --------------------------------------------------------------------------- plots


def expected_files(out_dir: Path, names, suffix: str = "") -> list[Path]:
    return [out_dir / f"{n}{suffix}.{ext}" for n in names for ext in ("png", "pdf")]


def test_make_plots_all(results_dir: Path, tmp_path: Path) -> None:
    make_plots = load_script("make_plots")
    out = tmp_path / "figs"
    assert make_plots.main(["--all", "--results-dir", str(results_dir), "--out-dir", str(out)]) == 0
    files = expected_files(out, [n for names in FIGURES.values() for n in names])
    for f in files:
        assert f.exists() and f.stat().st_size > 1000, f
    assert len(list(out.iterdir())) == len(files)


def test_make_plots_ieee_width(results_dir: Path, tmp_path: Path) -> None:
    from PIL import Image

    make_plots = load_script("make_plots")
    out = tmp_path / "figs"
    make_plots.main(["--exp", "e4", "--ieee", "--results-dir", str(results_dir), "--out-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["e4_secoqc_bars_ieee.pdf", "e4_secoqc_bars_ieee.png"]
    width_px = Image.open(out / "e4_secoqc_bars_ieee.png").size[0]
    assert width_px == 3.5 * 200  # exactly one IEEE column at 200 dpi


def test_make_plots_skips_missing_experiments(results_dir: Path, tmp_path: Path) -> None:
    partial = tmp_path / "partial"  # only E4 has results
    _copy(results_dir / "e4_secoqc", partial / "e4_secoqc")
    make_plots = load_script("make_plots")
    out = tmp_path / "figs"
    make_plots.main(["--all", "--results-dir", str(partial), "--out-dir", str(out)])
    assert sorted(p.stem for p in out.iterdir()) == ["e4_secoqc_bars", "e4_secoqc_bars"]
    with pytest.raises(FileNotFoundError, match="run_experiment.py"):
        make_plots.main(["--exp", "e1", "--results-dir", str(partial), "--out-dir", str(out)])


def _copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())


# --------------------------------------------------------------------------- tables


def test_make_tables(results_dir: Path, tmp_path: Path) -> None:
    make_tables = load_script("make_tables")
    out = tmp_path / "tables.md"
    assert make_tables.main(["--results-dir", str(results_dir), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for heading in ("## E2 mesh diagnostics", "## E2: SPR vs KAR", "### mesh0", "### diverse",
                    "## E3: mesh density", "## E3b: mesh density at a fixed arrival rate",
                    "## E1: chain", "## E4: SECOQC"):
        assert heading in text
    assert "Fixed arrival rate lambda = 95.1826 req/s" in text.split("## E3b")[1]
    # E1 KAR sanity check reports SPR's same-seed values next to KAR's.
    assert "rho = 0.5, seed 0: KAR vs SPR (same seed) outage" in text
    assert ": identical" in text.split("## E1")[1]
    assert "| mesh0 | 0 | 4 | 0.434 | yes | 0.156 | 4.07 |" in text
    assert "| diverse | 8 | 5 | 0.129 | no | 0.172 | 4.13 |" in text
    # One row per rho in each E2 mesh table, and one per degree in E3.
    e2_mesh0 = text.split("### mesh0")[1].split("###")[0]
    assert sum(line.startswith("| 0.") for line in e2_mesh0.splitlines()) == 3
    e3 = text.split("## E3: mesh density")[1].split("## E3b")[0]
    assert sum(line.startswith(("| 3 ", "| 4 ", "| 6 ")) for line in e3.splitlines()) == 3
    assert " ± " in text


def test_make_tables_notes_missing(tmp_path: Path) -> None:
    make_tables = load_script("make_tables")
    out = tmp_path / "tables.md"
    make_tables.main(["--results-dir", str(tmp_path / "nothing"), "--out", str(out)])
    assert "skipped" in out.read_text(encoding="utf-8")
