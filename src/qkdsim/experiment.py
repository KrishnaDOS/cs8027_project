"""Experiment runner: expand a sweep grid x seeds, run in parallel, resume, aggregate.

Experiment config (YAML), paths relative to the experiment file:

    name: e2_mesh_load
    base: ../../configs/default.yaml      # simulator config every run starts from
    overrides: {topology: {...}}          # nested; applied to every run
    seeds: [0, 1, 2, 3, 4]                # run seeds
    variants:                             # optional named override sets, crossed with grid
      - {label: mesh0, set: {topology.seed: 0}}
    grid:                                 # dotted key -> list of values (cartesian product)
      traffic.offered_load: [0.1, 0.2]
      routing.strategy: [spr, kar]
    extra_runs:                           # optional single points, per variant
      - {set: {traffic.offered_load: 0.5, routing.strategy: kar}, seeds: [0]}

Output (results/<name>/, or results/<name>_quick/ with --quick):
    runs/<run_id>.json   one file per run, written atomically as soon as the run finishes
    summary.csv          one row per run (grid parameters + metrics.summarize fields)
    by_hops.csv          per-run, per-hop-count breakdown
    meta.json            provenance: config hash, git commit, times, Python version

Resume: a run is skipped when its file exists and records the same config hash (the
hash of the fully resolved simulator config plus the seed). Changing the experiment
config therefore re-runs exactly the affected runs. Code changes do NOT invalidate
results; each run file records the git commit it was produced with.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from qkdsim.config import ConfigError, config_from_dict, deep_merge, read_yaml
from qkdsim.metrics import by_hops, summarize
from qkdsim.simulator import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

QUICK_DURATION_S = 300.0
QUICK_WARMUP_S = 60.0
QUICK_SEEDS = (0, 1, 2)

# Short names used in run ids; other keys use their last dotted component.
SHORT_KEYS = {
    "traffic.offered_load": "rho",
    "routing.strategy": "router",
    "topology.target_avg_degree": "deg",
    "topology.n_nodes": "n",
}
EXPERIMENT_KEYS = {"name", "base", "overrides", "seeds", "variants", "grid", "extra_runs", "description"}


@dataclass
class RunSpec:
    run_id: str
    variant: str | None
    params: dict[str, Any]  # dotted key -> value (variant + grid point / extra run)
    seed: int
    config: dict[str, Any]  # fully resolved, validated simulator config
    config_hash: str


@dataclass
class Experiment:
    name: str
    path: Path
    runs: list[RunSpec]
    quick: bool
    out_dir: Path
    file_sha256: str
    config_hash: str  # hash over all run hashes, in order
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- expansion


def nest_dotted(flat: Mapping[str, Any]) -> dict[str, Any]:
    """{"a.b": 1, "a.c": 2} -> {"a": {"b": 1, "c": 2}}."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        part: Any = value
        for name in reversed(key.split(".")):
            part = {name: part}
        nested = deep_merge(nested, part)
    return nested


def _fmt(value: Any) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in text)


def make_run_id(variant: str | None, point: Mapping[str, Any], seed: int) -> str:
    parts = [variant] if variant else []
    parts += [f"{SHORT_KEYS.get(k, k.split('.')[-1])}={_fmt(v)}" for k, v in point.items()]
    parts.append(f"seed={seed}")
    return "__".join(parts)


def run_hash(config: Mapping[str, Any], seed: int) -> str:
    payload = json.dumps({"config": config, "seed": seed}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_experiment(path: str | Path, quick: bool = False,
                    results_dir: str | Path = DEFAULT_RESULTS_DIR) -> Experiment:
    """Read an experiment config and expand it into RunSpecs (validating every config)."""
    path = Path(path).resolve()
    raw = read_yaml(path)
    unknown = sorted(set(raw) - EXPERIMENT_KEYS)
    if unknown:
        raise ConfigError(f"{path.name}: unknown experiment key(s) {unknown}")
    if "name" not in raw or "base" not in raw:
        raise ConfigError(f"{path.name}: experiment needs 'name' and 'base'")

    base = read_yaml((path.parent / raw["base"]).resolve())
    common = deep_merge(base, raw.get("overrides") or {})
    seeds: list[int] = list(raw.get("seeds") or common.get("simulation", {}).get("seeds", [0]))
    if quick:
        common = deep_merge(common, {"simulation": {"duration_s": QUICK_DURATION_S,
                                                    "warmup_s": QUICK_WARMUP_S}})
        seeds = list(QUICK_SEEDS)

    variants = raw.get("variants") or [{"label": None, "set": {}}]
    grid: dict[str, list[Any]] = raw.get("grid") or {}
    grid_points = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]

    points: list[tuple[dict[str, Any], list[int]]] = [(p, seeds) for p in grid_points]
    for extra in raw.get("extra_runs") or []:
        extra_seeds = list(extra.get("seeds", seeds))
        if quick:
            extra_seeds = [s for s in extra_seeds if s in QUICK_SEEDS] or [QUICK_SEEDS[0]]
        points.append((dict(extra["set"]), extra_seeds))

    runs: list[RunSpec] = []
    seen: set[str] = set()
    for variant in variants:
        label, variant_set = variant.get("label"), dict(variant.get("set") or {})
        for point, point_seeds in points:
            merged = deep_merge(common, nest_dotted({**variant_set, **point}))
            try:
                config = config_from_dict(merged).to_dict()
            except ConfigError as exc:
                raise ConfigError(f"{path.name}: point {label} {point}: {exc}") from None
            for seed in point_seeds:
                run_id = make_run_id(label, point, seed)
                if run_id in seen:
                    raise ConfigError(f"{path.name}: duplicate run id {run_id}")
                seen.add(run_id)
                runs.append(RunSpec(run_id, label, {**variant_set, **point}, int(seed), config,
                                    run_hash(config, int(seed))))

    name = raw["name"] + ("_quick" if quick else "")
    all_hash = hashlib.sha256("".join(r.config_hash for r in runs).encode()).hexdigest()
    return Experiment(
        name=name, path=path, runs=runs, quick=quick, out_dir=Path(results_dir) / name,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config_hash=all_hash, raw=raw,
    )


# --------------------------------------------------------------------------- execution


def git_state() -> tuple[str | None, bool | None]:
    """(commit hash, working tree dirty) or (None, None) if git is unavailable."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=REPO_ROOT, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / f"{run_id}.json"


def is_complete(path: Path, config_hash: str) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("config_hash") == config_hash
    except (OSError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def execute_run(task: dict[str, Any]) -> dict[str, Any]:
    """Run one simulation and write its result file atomically. Top-level so it can be
    pickled into worker processes."""
    cfg = config_from_dict(task["config"])
    t0 = time.perf_counter()
    result = run_simulation(cfg, task["seed"])
    wall_s = time.perf_counter() - t0
    record = {
        "run_id": task["run_id"],
        "experiment": task["experiment"],
        "variant": task["variant"],
        "params": task["params"],
        "seed": task["seed"],
        "config_hash": task["config_hash"],
        "git_commit": task["git_commit"],
        "wall_s": wall_s,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summarize(result, cfg),
        "by_hops": by_hops(result, cfg).to_dict(orient="records"),
    }
    path = Path(task["path"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(record), indent=1), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result
    return {"run_id": task["run_id"], "wall_s": wall_s}


def _cost_key(spec: RunSpec) -> tuple:
    """Rough cost ordering: KAR > SPR, then load, size, degree, simulated time."""
    c = spec.config
    return (c["routing"]["strategy"] == "kar", c["traffic"]["offered_load"] or 0.0,
            c["topology"]["n_nodes"], c["topology"]["target_avg_degree"],
            c["simulation"]["duration_s"])


def run_experiment(exp: Experiment, workers: int, log=print) -> dict[str, Any]:
    """Execute every not-yet-complete run of `exp`, then aggregate. Returns counts."""
    runs_dir = exp.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    commit, dirty = git_state()

    pending = [r for r in exp.runs if not is_complete(run_path(exp.out_dir, r.run_id), r.config_hash)]
    n_skipped = len(exp.runs) - len(pending)
    log(f"{exp.name}: {len(exp.runs)} runs, {n_skipped} already complete, {len(pending)} to run "
        f"({workers} worker{'s' if workers != 1 else ''})")
    pending.sort(key=_cost_key, reverse=True)  # longest first: shorter makespan
    tasks = [{
        "run_id": r.run_id, "experiment": exp.name, "variant": r.variant, "params": r.params,
        "seed": r.seed, "config": r.config, "config_hash": r.config_hash, "git_commit": commit,
        "path": str(run_path(exp.out_dir, r.run_id)),
    } for r in pending]

    failed: list[dict[str, str]] = []
    t0 = time.perf_counter()

    def report(i: int, done: dict[str, Any]) -> None:
        log(f"  [{i}/{len(tasks)}] {done['run_id']}  {done['wall_s']:.1f} s  "
            f"(elapsed {time.perf_counter() - t0:.0f} s)")

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                report(i, execute_run(task))
            except Exception as exc:  # keep going; failures are listed in meta.json
                failed.append({"run_id": task["run_id"], "error": repr(exc)})
                log(f"  FAILED {task['run_id']}: {exc!r}")
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute_run, task): task["run_id"] for task in tasks}
            for i, future in enumerate(cf.as_completed(futures), 1):
                try:
                    report(i, future.result())
                except Exception as exc:
                    failed.append({"run_id": futures[future], "error": repr(exc)})
                    log(f"  FAILED {futures[future]}: {exc!r}")

    n_rows = aggregate(exp)
    end = datetime.now(timezone.utc)
    meta_path = exp.out_dir / "meta.json"
    history = []
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            history = old.get("history", []) + [{k: old.get(k) for k in
                                                 ("start_time", "end_time", "git_commit", "n_executed")}]
        except ValueError:
            pass
    meta = {
        "experiment": exp.name,
        "config_file": str(exp.path),
        "config_file_sha256": exp.file_sha256,
        "config_hash": exp.config_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "quick": exp.quick,
        "workers": workers,
        "n_runs": len(exp.runs),
        "n_executed": len(tasks) - len(failed),
        "n_skipped": n_skipped,
        "n_failed": len(failed),
        "failed": failed,
        "n_summary_rows": n_rows,
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{exp.name}: done in {(end - start).total_seconds():.0f} s; {n_rows}/{len(exp.runs)} runs "
        f"in summary.csv; {len(failed)} failed -> {exp.out_dir}")
    return {"executed": len(tasks) - len(failed), "skipped": n_skipped, "failed": len(failed),
            "rows": n_rows}


def aggregate(exp: Experiment) -> int:
    """Write summary.csv and by_hops.csv from the completed run files of this expansion."""
    rows, hop_rows = [], []
    for spec in exp.runs:
        path = run_path(exp.out_dir, spec.run_id)
        if not is_complete(path, spec.config_hash):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        ident = {"run_id": spec.run_id, "variant": spec.variant, **spec.params, "seed": spec.seed}
        rows.append({**ident, "wall_s": rec["wall_s"], **rec["summary"]})
        hop_rows += [{**ident, **h} for h in rec["by_hops"]]
    pd.DataFrame(rows).to_csv(exp.out_dir / "summary.csv", index=False)
    pd.DataFrame(hop_rows).to_csv(exp.out_dir / "by_hops.csv", index=False)
    return len(rows)


# --------------------------------------------------------------------------- dry run


def dry_run(path: str | Path, workers: int, results_dir: Path, log=print) -> dict[str, float]:
    """Time the most expensive run and project the sweep's wall-clock (full and quick)."""
    full = load_experiment(path, quick=False, results_dir=results_dir)
    quick = load_experiment(path, quick=True, results_dir=results_dir)
    # Most expensive point = highest rho with KAR. If the sweep's highest-load point is
    # SPR-only (e.g. E1, whose single KAR run sits at a lower rho), time that one too.
    candidates = [max(full.runs, key=_cost_key)]
    heaviest_spr = max((r for r in full.runs if r.config["routing"]["strategy"] == "spr"),
                       key=_cost_key, default=None)
    if heaviest_spr is not None and _cost_key(heaviest_spr)[1:] > _cost_key(candidates[0])[1:]:
        candidates.append(heaviest_spr)
    t_max, cfg = 0.0, None
    for spec in candidates:
        log(f"{full.name}: timing {spec.run_id} ...")
        spec_cfg = config_from_dict(spec.config)
        t0 = time.perf_counter()
        run_simulation(spec_cfg, spec.seed)
        elapsed = time.perf_counter() - t0
        log(f"  {elapsed:.1f} s")
        if elapsed > t_max:
            t_max, cfg = elapsed, spec_cfg
    t_quick = t_max * QUICK_DURATION_S / cfg.simulation.duration_s  # cost ~ simulated time

    def projection(n_runs: int, t_run: float) -> tuple[float, float]:
        return n_runs * t_run, math.ceil(n_runs / workers) * t_run

    serial_full, par_full = projection(len(full.runs), t_max)
    serial_quick, par_quick = projection(len(quick.runs), t_quick)
    log(f"  slowest run: {t_max:.1f} s (full, {cfg.simulation.duration_s:g} s simulated); "
        f"~{t_quick:.1f} s at quick length")
    log(f"  full : {len(full.runs):4d} runs -> <= {par_full / 60:.1f} min with {workers} workers "
        f"(serial <= {serial_full / 60:.1f} min)")
    log(f"  quick: {len(quick.runs):4d} runs -> <= {par_quick / 60:.1f} min with {workers} workers "
        f"(serial <= {serial_quick / 60:.1f} min)")
    log("  (upper bounds: every run is costed as the most expensive one)")
    return {"t_max_s": t_max, "full_parallel_s": par_full, "quick_parallel_s": par_quick,
            "n_full": len(full.runs), "n_quick": len(quick.runs)}


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python experiments/run_experiment.py",
                                     description="Run a qkdsim experiment sweep.")
    parser.add_argument("config", help="experiment YAML (experiments/configs/*.yaml)")
    parser.add_argument("--quick", action="store_true",
                        help=f"duration {QUICK_DURATION_S:g} s, warmup {QUICK_WARMUP_S:g} s, "
                             f"seeds {list(QUICK_SEEDS)}; writes results/<exp>_quick/")
    parser.add_argument("--dry-run", action="store_true",
                        help="time the most expensive run, project total wall-clock, exit")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)

    if args.dry_run:
        dry_run(args.config, args.workers, Path(args.results_dir))
        return 0
    exp = load_experiment(args.config, quick=args.quick, results_dir=args.results_dir)
    counts = run_experiment(exp, args.workers)
    return 1 if counts["failed"] else 0
