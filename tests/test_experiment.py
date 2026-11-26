"""Tests for the experiment runner (Milestone 4) and mesh selection."""

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from qkdsim.config import ConfigError, load_config
from qkdsim.experiment import (
    QUICK_DURATION_S,
    QUICK_SEEDS,
    QUICK_WARMUP_S,
    dry_run,
    load_experiment,
    nest_dotted,
    run_experiment,
    run_path,
    source_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "experiments" / "configs"
DEFAULT_YAML = (REPO_ROOT / "configs" / "default.yaml").as_posix()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tiny_experiment(tmp_path: Path, extra: str = "") -> Path:
    """2 rho x 2 routers x 2 seeds on a 5-node chain, 30 s simulated: 8 fast runs."""
    path = tmp_path / "tiny.yaml"
    path.write_text(f"""
name: tiny
base: {DEFAULT_YAML}
overrides:
  topology: {{type: chain, n_nodes: 5}}
  simulation: {{duration_s: 30, warmup_s: 5}}
seeds: [0, 1]
grid:
  traffic.offered_load: [0.1, 0.5]
  routing.strategy: [spr, kar]
{extra}""", encoding="utf-8")
    return path


def quiet(*_args, **_kwargs) -> None:
    pass


# --------------------------------------------------------------------------- expansion


@pytest.mark.parametrize(("name", "n_full", "n_quick"), [
    ("e1_chain_hops", 3 * 5 + 1, 3 * 3 + 1),      # SPR grid + one KAR sanity run (seed 0)
    ("e2_mesh_load", 2 * 9 * 2 * 5, 2 * 9 * 2 * 3),  # meshes x rho x routers x seeds
    ("e3_mesh_density", 4 * 2 * 20, 4 * 2 * 3),    # degrees x routers x 20 topology seeds
    ("e3b_mesh_density_fixed_lambda", 4 * 2 * 20, 4 * 2 * 3),
    ("e4_secoqc", 4 * 2 * 5, 4 * 2 * 3),           # rho x routers x seeds
])
def test_grid_expansion_counts(name: str, n_full: int, n_quick: int, tmp_path: Path) -> None:
    full = load_experiment(CONFIGS / f"{name}.yaml", results_dir=tmp_path)
    quick = load_experiment(CONFIGS / f"{name}.yaml", quick=True, results_dir=tmp_path)
    assert len(full.runs) == n_full
    assert len(quick.runs) == n_quick
    assert len({r.run_id for r in full.runs}) == n_full  # unique ids


def test_quick_overrides(tmp_path: Path) -> None:
    full = load_experiment(CONFIGS / "e2_mesh_load.yaml", results_dir=tmp_path)
    quick = load_experiment(CONFIGS / "e2_mesh_load.yaml", quick=True, results_dir=tmp_path)
    assert quick.out_dir == tmp_path / "e2_mesh_load_quick"
    assert full.out_dir == tmp_path / "e2_mesh_load"
    assert {r.seed for r in quick.runs} == set(QUICK_SEEDS)
    assert {r.seed for r in full.runs} == {0, 1, 2, 3, 4}
    for r in quick.runs:
        assert r.config["simulation"]["duration_s"] == QUICK_DURATION_S
        assert r.config["simulation"]["warmup_s"] == QUICK_WARMUP_S
    assert {r.config["simulation"]["duration_s"] for r in full.runs} == {600}
    # Quick and full results never share a hash, so they can never be mixed up.
    assert not {r.config_hash for r in quick.runs} & {r.config_hash for r in full.runs}


def test_e1_kar_sanity_run_and_quick_seed(tmp_path: Path) -> None:
    for quick in (False, True):
        exp = load_experiment(CONFIGS / "e1_chain_hops.yaml", quick=quick, results_dir=tmp_path)
        kar = [r for r in exp.runs if r.config["routing"]["strategy"] == "kar"]
        assert [(r.params["traffic.offered_load"], r.seed) for r in kar] == [(0.8, 0)]
        assert all(r.config["topology"]["type"] == "chain" for r in exp.runs)
        assert {r.config["traffic"]["offered_load"] for r in exp.runs} == {0.65, 0.8, 0.95}


def test_e3b_fixes_lambda_on_e3_grid(tmp_path: Path) -> None:
    e3 = load_experiment(CONFIGS / "e3_mesh_density.yaml", results_dir=tmp_path)
    e3b = load_experiment(CONFIGS / "e3b_mesh_density_fixed_lambda.yaml", results_dir=tmp_path)
    assert [r.run_id for r in e3b.runs] == [r.run_id for r in e3.runs]  # same grid and seeds
    for r in e3b.runs:
        assert r.config["traffic"]["offered_load"] is None
        assert r.config["traffic"]["arrival_rate_per_s"] == pytest.approx(95.1826)
    for a, b in zip(e3.runs, e3b.runs):  # identical topologies: only the traffic differs
        assert a.config["topology"] == b.config["topology"]
    assert "95.182639" in (CONFIGS / "e3b_mesh_density_fixed_lambda.yaml").read_text()  # derivation kept


def test_e2_meshes_and_e3_topology_seeds(tmp_path: Path) -> None:
    e2 = load_experiment(CONFIGS / "e2_mesh_load.yaml", results_dir=tmp_path)
    assert {(r.variant, r.config["topology"]["seed"]) for r in e2.runs} == {("mesh0", 0), ("diverse", 8)}
    e3 = load_experiment(CONFIGS / "e3_mesh_density.yaml", results_dir=tmp_path)
    assert {r.config["topology"]["seed"] for r in e3.runs} == {None}  # topology follows run seed
    assert {r.config["traffic"]["offered_load"] for r in e3.runs} == {0.3}


def test_e2_diverse_seed_matches_selection_rule() -> None:
    select_mesh = load_script("select_mesh")
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    rows = select_mesh.scan(cfg, range(9))  # enough to confirm the LOWEST qualifying seed
    assert select_mesh.select_seed(rows) == 8
    assert rows[8]["max_bridge_share"] <= 0.15
    assert rows[0]["bottleneck_is_bridge"]


def test_demo_config_matches_e2_diverse_mesh(tmp_path: Path) -> None:
    demo = load_config(CONFIGS / "demo_diverse.yaml")
    assert (demo.simulation.duration_s, demo.simulation.warmup_s) == (120, 20)
    assert demo.traffic.offered_load == 0.3
    e2 = load_experiment(CONFIGS / "e2_mesh_load.yaml", results_dir=tmp_path)
    diverse = next(r for r in e2.runs if r.variant == "diverse")
    assert demo.topology.seed == diverse.config["topology"]["seed"]
    # Same mesh as E2 apart from the topology seed being the only topology override.
    assert {k: v for k, v in demo.to_dict()["topology"].items()} == diverse.config["topology"]


def test_nest_dotted() -> None:
    assert nest_dotted({"a.b": 1, "a.c": 2, "d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}


def test_invalid_experiment_rejected(tmp_path: Path) -> None:
    bad = write_tiny_experiment(tmp_path, "colour: blue\n")
    with pytest.raises(ConfigError, match="unknown experiment key"):
        load_experiment(bad, results_dir=tmp_path)
    dup = write_tiny_experiment(tmp_path, "extra_runs:\n  - set: {traffic.offered_load: 0.1, routing.strategy: spr}\n")
    with pytest.raises(ConfigError, match="duplicate run id"):
        load_experiment(dup, results_dir=tmp_path)


# --------------------------------------------------------------------------- execution


def test_run_resume_and_aggregate(tmp_path: Path) -> None:
    exp = load_experiment(write_tiny_experiment(tmp_path), results_dir=tmp_path)
    counts = run_experiment(exp, workers=1, log=quiet)
    assert counts == {"executed": 8, "skipped": 0, "failed": 0, "rows": 8}
    files = sorted((exp.out_dir / "runs").glob("*.json"))
    assert len(files) == 8

    summary = pd.read_csv(exp.out_dir / "summary.csv")
    by_hops = pd.read_csv(exp.out_dir / "by_hops.csv")
    assert len(summary) == 8
    assert set(summary["routing.strategy"]) == {"spr", "kar"}
    expected_hop_rows = sum(len(json.loads(f.read_text())["by_hops"]) for f in files)
    assert len(by_hops) == expected_hop_rows > 8
    assert (by_hops.groupby("run_id")["count"].sum().sort_index()
            == summary.set_index("run_id")["offered"].sort_index()).all()

    meta = json.loads((exp.out_dir / "meta.json").read_text())
    for key in ("config_hash", "git_commit", "start_time", "end_time", "python_version"):
        assert key in meta
    assert meta["n_runs"] == 8

    # Resume: everything complete -> nothing runs.
    mtimes = {f.name: f.stat().st_mtime_ns for f in files}
    assert run_experiment(exp, workers=1, log=quiet)["executed"] == 0
    # Delete one result and corrupt another's hash: exactly those two re-run.
    run_path(exp.out_dir, exp.runs[0].run_id).unlink()
    stale = run_path(exp.out_dir, exp.runs[1].run_id)
    record = json.loads(stale.read_text())
    record["config_hash"] = "stale"
    stale.write_text(json.dumps(record))
    counts = run_experiment(exp, workers=1, log=quiet)
    assert counts == {"executed": 2, "skipped": 6, "failed": 0, "rows": 8}
    for f in files[2:]:
        if f.name not in {exp.runs[0].run_id + ".json", exp.runs[1].run_id + ".json"}:
            assert f.stat().st_mtime_ns == mtimes[f.name]  # untouched
    assert len(json.loads((exp.out_dir / "meta.json").read_text())["history"]) == 2


def test_aggregate_skips_runs_outside_current_grid(tmp_path: Path) -> None:
    path = write_tiny_experiment(tmp_path)
    exp = load_experiment(path, results_dir=tmp_path)
    run_experiment(exp, workers=1, log=quiet)
    # Shrink the grid (as when E1's rho values change): old files stay on disk.
    path.write_text(path.read_text().replace("[0.1, 0.5]", "[0.5]"), encoding="utf-8")
    smaller = load_experiment(path, results_dir=tmp_path)
    messages: list[str] = []
    counts = run_experiment(smaller, workers=1, log=messages.append)
    assert counts == {"executed": 0, "skipped": 4, "failed": 0, "rows": 4}
    warnings = [m for m in messages if m.startswith("WARNING")]
    assert len(warnings) == 1 and "4 run file(s)" in warnings[0] and "rho=0.1" in warnings[0]
    summary = pd.read_csv(smaller.out_dir / "summary.csv")
    assert set(summary["traffic.offered_load"]) == {0.5}
    assert set(pd.read_csv(smaller.out_dir / "by_hops.csv")["traffic.offered_load"]) == {0.5}
    assert len(list((smaller.out_dir / "runs").glob("*.json"))) == 8  # nothing deleted


def copy_package(tmp_path: Path) -> Path:
    """A copy of src/qkdsim/ to edit, so the tests never touch the real sources."""
    src = REPO_ROOT / "src" / "qkdsim"
    dst = tmp_path / "qkdsim_copy"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def copy_data(tmp_path: Path) -> Path:
    """A copy of data/*.yaml to edit (the real data/secoqc.yaml is only read)."""
    dst = tmp_path / "data_copy"
    dst.mkdir()
    for f in (REPO_ROOT / "data").glob("*.yaml"):
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def append_line(path: Path, line: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + f"\n{line}\n", encoding="utf-8")


def test_source_hash_defaults_and_normalization(tmp_path: Path) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    base = source_hash(pkg, data)
    assert base == source_hash(pkg, data)  # deterministic
    # Defaults are the real package and the real data folder.
    assert source_hash() == source_hash(REPO_ROOT / "src" / "qkdsim", REPO_ROOT / "data") == base
    # Line endings do not matter (CRLF vs LF checkouts hash the same), for code and data.
    for f in (pkg / "simulator.py", data / "secoqc.yaml"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert source_hash(pkg, data) == base
    # __pycache__ is ignored.
    (pkg / "__pycache__").mkdir(exist_ok=True)
    (pkg / "__pycache__" / "x.py").write_text("junk", encoding="utf-8")
    assert source_hash(pkg, data) == base


@pytest.mark.parametrize("module", ["analysis.py", "experiment.py", "run.py"])
def test_editing_excluded_module_keeps_hash(tmp_path: Path, module: str) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    base = source_hash(pkg, data)
    append_line(pkg / module, "# plotting / runner / CLI edit")
    assert source_hash(pkg, data) == base


@pytest.mark.parametrize("module", ["simulator.py", "links.py", "routing.py", "traffic.py",
                                    "topologies.py", "metrics.py", "config.py", "__init__.py"])
def test_editing_simulation_module_changes_hash(tmp_path: Path, module: str) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    base = source_hash(pkg, data)
    append_line(pkg / module, "# simulation edit")
    assert source_hash(pkg, data) != base


def test_adding_new_module_changes_hash(tmp_path: Path) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    base = source_hash(pkg, data)
    (pkg / "new_module.py").write_text("X = 1\n", encoding="utf-8")
    with_module = source_hash(pkg, data)
    assert with_module != base
    (pkg / "sub").mkdir()  # modules in subpackages count too
    (pkg / "sub" / "deeper.py").write_text("Y = 2\n", encoding="utf-8")
    assert source_hash(pkg, data) not in (base, with_module)


def test_data_yaml_edits_and_additions_change_hash(tmp_path: Path) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    base = source_hash(pkg, data)
    append_line(data / "secoqc.yaml", "# edited link note")
    edited = source_hash(pkg, data)
    assert edited != base
    (data / "new_topology.yaml").write_text("name: x\n", encoding="utf-8")
    added = source_hash(pkg, data)
    assert added not in (base, edited)
    (data / "notes.txt").write_text("not topology data", encoding="utf-8")  # only *.yaml counts
    assert source_hash(pkg, data) == added


def test_code_change_invalidates_cache_and_unchanged_code_resumes(tmp_path: Path) -> None:
    pkg, data = copy_package(tmp_path), copy_data(tmp_path)
    path = write_tiny_experiment(tmp_path)

    def load():
        return load_experiment(path, results_dir=tmp_path, source_dir=pkg, data_dir=data)

    exp = load()
    assert run_experiment(exp, workers=1, log=quiet)["executed"] == 8
    meta = json.loads((exp.out_dir / "meta.json").read_text())
    assert meta["code_hash"] == exp.code_hash == source_hash(pkg, data)
    record = json.loads(run_path(exp.out_dir, exp.runs[0].run_id).read_text())
    assert record["code_hash"] == exp.code_hash

    # Unchanged code (and config): everything resumes.
    same = load()
    assert [r.config_hash for r in same.runs] == [r.config_hash for r in exp.runs]
    assert run_experiment(same, workers=1, log=quiet) == {"executed": 0, "skipped": 8, "failed": 0, "rows": 8}

    # Editing the plotting helpers does not invalidate anything.
    append_line(pkg / "analysis.py", "# new plot helper")
    assert run_experiment(load(), workers=1, log=quiet)["executed"] == 0

    # Edit one simulator source file: every cached run is stale and re-runs.
    append_line(pkg / "simulator.py", "# changed")
    changed = load()
    assert changed.code_hash != exp.code_hash
    assert not {r.config_hash for r in changed.runs} & {r.config_hash for r in exp.runs}
    assert run_experiment(changed, workers=1, log=quiet) == {"executed": 8, "skipped": 0, "failed": 0, "rows": 8}
    meta = json.loads((exp.out_dir / "meta.json").read_text())
    assert meta["code_hash"] == changed.code_hash
    assert [h["code_hash"] for h in meta["history"]] == [exp.code_hash] * 3
    # ...and with the changed code left in place, the next invocation resumes again.
    assert run_experiment(load(), workers=1, log=quiet)["executed"] == 0

    # Editing topology data also re-runs everything.
    append_line(data / "secoqc.yaml", "# corrected link length")
    assert run_experiment(load(), workers=1, log=quiet)["executed"] == 8


def test_parallel_matches_serial(tmp_path: Path) -> None:
    path = write_tiny_experiment(tmp_path)
    serial = load_experiment(path, results_dir=tmp_path / "serial")
    parallel = load_experiment(path, results_dir=tmp_path / "parallel")
    run_experiment(serial, workers=1, log=quiet)
    run_experiment(parallel, workers=2, log=quiet)
    a = pd.read_csv(serial.out_dir / "summary.csv").drop(columns=["wall_s"])
    b = pd.read_csv(parallel.out_dir / "summary.csv").drop(columns=["wall_s"])
    pd.testing.assert_frame_equal(a, b)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    lines: list[str] = []
    out = dry_run(write_tiny_experiment(tmp_path), workers=4, results_dir=tmp_path / "res",
                  log=lines.append)
    assert out["n_full"] == 8 and out["n_quick"] == 2 * 2 * 3
    assert out["full_parallel_s"] == pytest.approx(2 * out["t_max_s"])  # ceil(8 / 4) waves
    assert "rho=0.5__router=kar" in lines[0]  # most expensive point: highest rho, KAR
    assert not any("router=spr" in line for line in lines)  # KAR already at the top rho
    assert not (tmp_path / "res").exists()


def test_dry_run_also_times_heavier_spr_point(tmp_path: Path) -> None:
    # Like E1: the only KAR run sits below the sweep's highest rho.
    path = write_tiny_experiment(tmp_path).read_text().replace("routing.strategy: [spr, kar]",
                                                               "routing.strategy: [spr]")
    path += "extra_runs:\n  - {set: {traffic.offered_load: 0.1, routing.strategy: kar}, seeds: [0]}\n"
    (tmp_path / "e1like.yaml").write_text(path, encoding="utf-8")
    lines: list[str] = []
    dry_run(tmp_path / "e1like.yaml", workers=2, results_dir=tmp_path / "res", log=lines.append)
    timed = [line for line in lines if "timing" in line]
    assert len(timed) == 2
    assert "rho=0.1__router=kar" in timed[0] and "rho=0.5__router=spr" in timed[1]
