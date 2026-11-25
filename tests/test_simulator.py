"""Tests for the SimPy engine, metrics and the CLI (Milestone 2)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qkdsim.config import Config, ConfigError, config_from_dict, deep_merge, load_config
from qkdsim.metrics import by_hops, measured_records, summarize
from qkdsim.run import main, parse_set_args
from qkdsim.simulator import OUTCOMES, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small, fast scenario: 12-node mesh, 100 s simulated, 20 s warmup.
BASE: dict[str, Any] = {
    "simulation": {"duration_s": 100, "warmup_s": 20},
    "topology": {"type": "random_mesh", "n_nodes": 12, "area_km": 50, "target_avg_degree": 4},
}


def make_cfg(**sections: dict[str, Any]) -> Config:
    return config_from_dict(deep_merge(BASE, sections))


@pytest.fixture(scope="module")
def loaded_run():
    """One overloaded run shared by several tests (small pools so they drain quickly; a
    short timeout keeps the number of retries, and so the test time, down)."""
    cfg = make_cfg(traffic={"offered_load": 1.5}, links={"pool_capacity_bits": 20000},
                   relay={"timeout_s": 1.0})
    return cfg, run_simulation(cfg, seed=3)


# --------------------------------------------------------------------------- conservation


def test_conservation_per_link(loaded_run) -> None:
    cfg, result = loaded_run
    links = result.links
    assert (links["consumed_bits"] > 0).all()
    assert (links["consumed_bits"] <= links["initial_bits"] + links["generated_bits"] + 1e-6).all()
    # Exact bookkeeping: pool = initial + generated - consumed; generated + discarded = R * T.
    np.testing.assert_allclose(
        links["final_bits"], links["initial_bits"] + links["generated_bits"] - links["consumed_bits"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        links["generated_bits"] + links["discarded_bits"], links["rate_bps"] * result.end_time_s,
        rtol=1e-9, atol=1e-3,
    )
    # Every delivered request debited K on each hop.
    delivered = result.records[result.records["outcome"] == "delivered"]
    k = cfg.traffic.key_size_bits
    assert links["consumed_bits"].sum() == pytest.approx(k * delivered["hops"].sum())


# --------------------------------------------------------------------------- load regimes


def test_low_load_no_outage_and_latency_matches_a6() -> None:
    cfg = make_cfg(traffic={"offered_load": 0.05})
    result = run_simulation(cfg, seed=0)
    summary = summarize(result, cfg)
    assert summary["offered"] > 100
    assert summary["outage_probability"] <= 0.001
    delivered = measured_records(result, cfg)
    delivered = delivered[delivered["outcome"] == "delivered"]
    relay = cfg.relay
    expected_s = np.mean(delivered["hops"] * relay.per_hop_processing_s
                         + delivered["path_length_km"] * relay.propagation_s_per_km)
    assert summary["latency_mean_s"] == pytest.approx(expected_s, rel=0.05)


def test_high_load_has_outages(loaded_run) -> None:
    cfg, result = loaded_run
    summary = summarize(result, cfg)
    assert summary["offered_load"] == pytest.approx(1.5)
    assert summary["outage_probability"] > 0
    outages = result.records[result.records["outcome"] == "outage"]
    # An outage resolves exactly at its deadline.
    np.testing.assert_allclose(outages["wait_s"], cfg.relay.timeout_s, rtol=1e-9)
    assert outages["latency_s"].isna().all()


def test_latency_formula_per_request(loaded_run) -> None:
    cfg, result = loaded_run
    d = result.records[result.records["outcome"] == "delivered"]
    relay = cfg.relay
    expected = d["wait_s"] + d["hops"] * relay.per_hop_processing_s + d["path_length_km"] * relay.propagation_s_per_km
    np.testing.assert_allclose(d["latency_s"], expected)
    assert (d["wait_s"] <= cfg.relay.timeout_s + 1e-9).all()


# --------------------------------------------------------------------------- determinism


def test_same_seed_identical_different_seed_differs() -> None:
    cfg = make_cfg(traffic={"offered_load": 0.8})
    a, b, c = run_simulation(cfg, 1), run_simulation(cfg, 1), run_simulation(cfg, 2)
    pd.testing.assert_frame_equal(a.records, b.records)
    pd.testing.assert_frame_equal(a.timeseries, b.timeseries)
    assert summaries_equal(summarize(a, cfg), summarize(b, cfg))
    assert not summaries_equal(summarize(a, cfg), summarize(c, cfg))


def summaries_equal(x: dict, y: dict) -> bool:
    """Equality that treats NaN == NaN (e.g. latency stats with no deliveries)."""
    def same(p, q) -> bool:
        return p == q or (isinstance(p, float) and isinstance(q, float) and np.isnan(p) and np.isnan(q))
    return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)


# --------------------------------------------------------------------------- run lifecycle


def test_warmup_excluded_from_metrics(loaded_run) -> None:
    cfg, result = loaded_run
    r = result.records
    assert (r["arrival_s"] < cfg.simulation.warmup_s).any()  # warmup arrivals were simulated...
    in_window = r[(r["arrival_s"] >= 20) & (r["arrival_s"] < 100)]
    summary = summarize(result, cfg)
    assert summary["offered"] == len(in_window) < len(r)  # ...but not counted
    assert summary["delivered"] + summary["outage"] + summary["no_path"] == summary["offered"]
    assert by_hops(result, cfg)["count"].sum() == summary["offered"]
    assert summary["delivered_throughput_bps"] == pytest.approx(
        summary["delivered"] * cfg.traffic.key_size_bits / 80.0
    )


def test_no_unresolved_requests_at_end(loaded_run) -> None:
    cfg, result = loaded_run
    r = result.records
    assert len(r) == result.n_spawned
    assert r["request_id"].tolist() == list(range(result.n_spawned))
    assert set(r["outcome"]) <= set(OUTCOMES)
    assert r["arrival_s"].max() < cfg.simulation.duration_s
    assert (r["resolved_s"] <= r["arrival_s"] + cfg.relay.timeout_s + 1e-9).all()
    # Requests arriving just before duration_s were still followed to resolution.
    assert result.end_time_s > cfg.simulation.duration_s
    assert result.end_time_s <= cfg.simulation.duration_s + cfg.relay.timeout_s + 1e-9


def test_sampler_timeseries(loaded_run) -> None:
    cfg, result = loaded_run
    ts = result.timeseries
    np.testing.assert_allclose(ts["time_s"], np.arange(0, 100, 1.0))
    assert ts["mean_fill_fraction"].between(0, 1).all()
    assert ts["frac_links_below_k"].between(0, 1).all()
    assert ts["mean_fill_fraction"].iloc[0] == pytest.approx(0.5)  # initial_fill_fraction
    assert ts["frac_links_below_k"].max() > 0  # overload drains some links


def test_key_larger_than_capacity_raises() -> None:
    cfg = make_cfg(traffic={"key_size_bits": 4096}, links={"pool_capacity_bits": 2048})
    with pytest.raises(ConfigError, match="key_size_bits=4096 exceeds"):
        run_simulation(cfg, 0)


def test_single_saturating_flow_on_chain_hits_link_rate() -> None:
    # Addendum A.4: an end-to-end flow is limited by its bottleneck link (all 5000 bps here).
    # Fixed-pair rho is normalized by the path's bottleneck rate, so rho=2 is 2x demand.
    cfg = make_cfg(
        topology={"type": "chain", "n_nodes": 5},
        traffic={"pair_mode": "fixed", "fixed_pair": [0, 4], "offered_load": 2.0},
        simulation={"duration_s": 200, "warmup_s": 60},  # initial pool drains in ~20 s
    )
    result = run_simulation(cfg, 0)
    assert result.lambda_per_s == pytest.approx(2 * 5000 / 256)
    summary = summarize(result, cfg)
    assert summary["delivered_throughput_bps"] == pytest.approx(5000, rel=0.03)
    assert summary["rho_bn"] == pytest.approx(2.0)
    assert summary["key_utilization"] == pytest.approx(1.0, rel=0.03)  # every link saturated


# --------------------------------------------------------------------------- topology seed


def test_topology_seed_fixes_graph_across_run_seeds() -> None:
    cfg = make_cfg(topology={"seed": 7}, traffic={"offered_load": 0.3})
    a, b = run_simulation(cfg, 1), run_simulation(cfg, 2)
    assert a.topology_seed == b.topology_seed == 7
    edges = ["u", "v", "length_km"]
    pd.testing.assert_frame_equal(a.links[edges], b.links[edges])  # same graph
    assert not a.records[["arrival_s", "src", "dst"]].equals(b.records[["arrival_s", "src", "dst"]])
    assert summarize(a, cfg)["topology_seed"] == 7


def test_topology_seed_unset_follows_run_seed() -> None:
    cfg = make_cfg(traffic={"offered_load": 0.3})
    a, b = run_simulation(cfg, 1), run_simulation(cfg, 2)
    assert (a.topology_seed, b.topology_seed) == (1, 2)
    assert not a.links[["u", "v", "length_km"]].equals(b.links[["u", "v", "length_km"]])
    # Fixing topology.seed to the run seed reproduces the unset behaviour exactly.
    same = run_simulation(make_cfg(topology={"seed": 1}, traffic={"offered_load": 0.3}), 1)
    pd.testing.assert_frame_equal(a.records, same.records)


# --------------------------------------------------------------------------- diagnostics


def test_bottleneck_diagnostics_in_summary(loaded_run) -> None:
    cfg, result = loaded_run
    s = summarize(result, cfg)
    assert s["rho_bn"] > 0
    assert s["rho_sat"] == pytest.approx(s["offered_load"] / s["rho_bn"])
    assert s["rho_sat"] < 1  # SPR concentrates load: some link saturates below rho=1


def test_key_utilization(loaded_run) -> None:
    cfg, result = loaded_run
    s = summarize(result, cfg)
    links = result.links
    window_s = cfg.simulation.duration_s - cfg.simulation.warmup_s
    # Produced key in the window is exactly rate * window length.
    np.testing.assert_allclose(links["window_produced_bits"], links["rate_bps"] * window_s, rtol=1e-9)
    assert s["key_utilization"] == pytest.approx(
        links["window_consumed_bits"].sum() / (links["rate_bps"].sum() * window_s)
    )
    assert 0 < s["link_utilization_min"] <= s["link_utilization_median"] <= s["link_utilization_max"]
    assert s["link_utilization_max"] <= 1.0 + cfg.links.pool_capacity_bits / (5000 * window_s)
    assert 0 < s["key_utilization"] < 1


def test_low_load_key_utilization_is_small() -> None:
    cfg = make_cfg(traffic={"offered_load": 0.05})
    s = summarize(run_simulation(cfg, 0), cfg)
    # rho = consumed / generation capacity on SPR paths, so utilization ~= rho at low load.
    assert s["key_utilization"] == pytest.approx(0.05, rel=0.25)


# --------------------------------------------------------------------------- KAR


def test_chain_spr_and_kar_identical_records() -> None:
    # Overloaded with small pools, so requests wait, retry and time out (KAR reads every
    # link's level on each retry; the chain leaves it only one path).
    base = {"topology": {"type": "chain", "n_nodes": 8}, "links": {"pool_capacity_bits": 20000},
            "traffic": {"offered_load": 1.5}, "relay": {"timeout_s": 2.0}}
    spr_cfg = make_cfg(**base)
    kar_cfg = make_cfg(**deep_merge(base, {"routing": {"strategy": "kar"}}))
    spr, kar = run_simulation(spr_cfg, 4), run_simulation(kar_cfg, 4)
    assert (spr.records["outcome"] == "outage").any()  # contention actually happened
    pd.testing.assert_frame_equal(spr.records, kar.records)
    pd.testing.assert_frame_equal(spr.timeseries, kar.timeseries)
    s_spr, s_kar = summarize(spr, spr_cfg), summarize(kar, kar_cfg)
    s_spr.pop("strategy"), s_kar.pop("strategy")
    assert summaries_equal(s_spr, s_kar)


def test_kar_low_load_default_mesh() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      overrides={"traffic": {"offered_load": 0.02}, "topology": {"seed": 0}})
    kar_cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                          overrides={"traffic": {"offered_load": 0.02}, "topology": {"seed": 0},
                                     "routing": {"strategy": "kar"}})
    spr = summarize(run_simulation(cfg, 0), cfg)
    kar = summarize(run_simulation(kar_cfg, 0), kar_cfg)
    assert kar["offered"] > 1000
    assert kar["outage_probability"] <= 0.001
    assert kar["mean_hops"] == pytest.approx(spr["mean_hops"], rel=0.05)


def test_secoqc_per_link_runs() -> None:
    cfg = make_cfg(topology={"type": "secoqc"}, links={"rate_model": "per_link"})
    with pytest.warns(UserWarning, match="ASSUMED"):
        result = run_simulation(cfg, 0)
    assert result.total_rate_bps == pytest.approx(700 + 3100 + 2500 + 15000 + 8000 + 1000 + 1000 + 1000)


# --------------------------------------------------------------------------- CLI


def test_parse_set_args() -> None:
    assert parse_set_args(["traffic.offered_load=1.0", "topology.type=chain", "simulation.seeds=[3, 4]",
                           "traffic.arrival_rate_per_s=null"]) == {
        "traffic": {"offered_load": 1.0, "arrival_rate_per_s": None},
        "topology": {"type": "chain"},
        "simulation": {"seeds": [3, 4]},
    }
    with pytest.raises(ConfigError):
        parse_set_args(["no_equals_sign"])


def test_cli_prints_summary(capsys) -> None:
    rc = main(["--config", str(REPO_ROOT / "configs" / "default.yaml"),
               "--set", "simulation.duration_s=30", "--set", "simulation.warmup_s=5",
               "--set", "topology.n_nodes=10", "--set", "traffic.offered_load=0.2", "--by-hops"])
    out = capsys.readouterr().out
    assert rc == 0
    for text in ("rho=0.2", "lambda=", "rho_bn=", "rho_sat=", "topology seed=", "outage_probability",
                 "delivered_throughput_bps", "latency_p95_s", "key_utilization",
                 "link_utilization_median", "by hop count", "wall-clock"):
        assert text in out
