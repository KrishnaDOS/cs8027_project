"""Tests for config dataclasses and YAML loading (Milestone 0)."""

from pathlib import Path

import pytest

from qkdsim.config import (
    Config,
    ConfigError,
    config_from_dict,
    deep_merge,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_default_yaml_matches_dataclass_defaults() -> None:
    assert load_config(DEFAULT_YAML) == Config()


def test_addendum_a1_defaults() -> None:
    cfg = load_config(DEFAULT_YAML)
    assert cfg.links.rate_model == "fixed"
    assert cfg.links.distance.ref_length_km == 25
    assert cfg.links.distance.ref_rate_bps == 5000
    assert cfg.topology.type == "random_mesh"
    assert cfg.topology.chain.spacing_km == pytest.approx(63.8)


@pytest.mark.parametrize("model", ["fixed", "distance", "per_link"])
def test_rate_model_accepts_all_values(model: str) -> None:
    cfg = load_config(DEFAULT_YAML, overrides={"links": {"rate_model": model}})
    assert cfg.links.rate_model == model


def test_rate_model_rejects_unknown_value() -> None:
    with pytest.raises(ConfigError, match="rate_model"):
        config_from_dict({"links": {"rate_model": "magic"}})


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"links": {"fixed_rate_bsp": 5000}})
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"nonsense": {}})


def test_partial_override_keeps_other_defaults() -> None:
    cfg = load_config(
        DEFAULT_YAML,
        overrides={"topology": {"type": "chain", "n_nodes": 32}, "links": {"distance": {"ref_length_km": 40}}},
    )
    assert cfg.topology.type == "chain"
    assert cfg.topology.n_nodes == 32
    assert cfg.topology.chain.spacing_km == pytest.approx(63.8)
    assert cfg.links.distance.ref_length_km == 40
    assert cfg.links.distance.ref_rate_bps == 5000  # sibling key untouched


def test_files_merge_in_order(tmp_path: Path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "topology:\n  type: chain\n  chain:\n    spacing_km: 65\nrouting:\n  strategy: kar\n",
        encoding="utf-8",
    )
    cfg = load_config(DEFAULT_YAML, exp)
    assert cfg.topology.type == "chain"
    assert cfg.topology.chain.spacing_km == 65
    assert cfg.routing.strategy == "kar"
    assert cfg.topology.n_nodes == 30  # from default.yaml


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"b": 10}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_offered_load_and_arrival_rate_exclusive() -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        config_from_dict({"traffic": {"arrival_rate_per_s": 2.0}})  # offered_load still 0.5
    with pytest.raises(ConfigError, match="exactly one"):
        config_from_dict({"traffic": {"offered_load": None}})
    cfg = config_from_dict({"traffic": {"arrival_rate_per_s": 2.0, "offered_load": None}})
    assert cfg.traffic.arrival_rate_per_s == 2.0


def test_fixed_pair_mode_requires_pair() -> None:
    with pytest.raises(ConfigError, match="fixed_pair"):
        config_from_dict({"traffic": {"pair_mode": "fixed"}})
    cfg = config_from_dict({"traffic": {"pair_mode": "fixed", "fixed_pair": [0, 5]}})
    assert cfg.traffic.fixed_pair == [0, 5]


def test_invalid_warmup_rejected() -> None:
    with pytest.raises(ConfigError, match="warmup_s"):
        config_from_dict({"simulation": {"duration_s": 100, "warmup_s": 100}})


def test_to_dict_roundtrip() -> None:
    cfg = load_config(DEFAULT_YAML)
    assert config_from_dict(cfg.to_dict()) == cfg
