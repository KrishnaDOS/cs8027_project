"""Configuration dataclasses and YAML loading/merging.

The dataclass defaults below are the single source of truth for default parameters;
configs/default.yaml mirrors them (a test checks that the two agree).

Loading works on plain dicts first and converts to dataclasses last:
    load_config(path_a, path_b, ...)  ->  deep-merge YAML files in order onto the defaults
                                      ->  Config (validated)
Unknown keys are rejected so that typos in experiment configs fail loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

RATE_MODELS = ("fixed", "distance", "per_link")
PAIR_MODES = ("uniform", "fixed")
ROUTING_STRATEGIES = ("spr", "kar")
TOPOLOGY_TYPES = ("chain", "secoqc", "random_mesh")


class ConfigError(ValueError):
    """Raised for unknown keys or invalid values in a configuration."""


# --------------------------------------------------------------------------- sections


@dataclass
class SimulationConfig:
    duration_s: float = 600.0
    warmup_s: float = 60.0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    sample_interval_s: float = 1.0


@dataclass
class DistanceRateConfig:
    """Parameters of the distance rate model R(L) = ref_rate_bps * 10^(-alpha*(L - ref_length_km)/10)."""

    ref_rate_bps: float = 5000.0
    # Peev 2009 Sec. 3.2: Toshiba link delivered 5.7 kbit/s over 25 km (Addendum A.1).
    ref_length_km: float = 25.0
    attenuation_db_per_km: float = 0.2


@dataclass
class LinksConfig:
    # fixed: every link gets fixed_rate_bps. distance: DistanceRateConfig.
    # per_link: skr_bps is read from the topology data file (SECOQC); implemented in M1+.
    rate_model: str = "fixed"
    fixed_rate_bps: float = 5000.0
    distance: DistanceRateConfig = field(default_factory=DistanceRateConfig)
    pool_capacity_bits: float = 200_000.0
    initial_fill_fraction: float = 0.5


@dataclass
class TrafficConfig:
    key_size_bits: int = 256
    pair_mode: str = "uniform"
    # Used only when pair_mode == "fixed": [src, dst] node ids.
    fixed_pair: list[Any] | None = None
    # Exactly one of these two must be set (the other None).
    arrival_rate_per_s: float | None = None
    offered_load: float | None = 0.5


@dataclass
class RelayConfig:
    timeout_s: float = 5.0
    per_hop_processing_s: float = 0.001
    propagation_s_per_km: float = 5.0e-6
    min_retry_interval_s: float = 0.001


@dataclass
class RoutingConfig:
    strategy: str = "spr"
    kar_epsilon_bits: float = 1.0


@dataclass
class ChainTopologyConfig:
    # Chen 2021 (Nature) Table 1: Beijing-Shanghai backbone, 32 trusted relays,
    # 31 links, average link length 63.8 km (Addendum A.1).
    spacing_km: float = 63.8


@dataclass
class TopologyConfig:
    type: str = "random_mesh"
    n_nodes: int = 30
    # random_mesh parameters
    area_km: float = 50.0
    target_avg_degree: float = 4.0
    # chain parameters (used when type == "chain")
    chain: ChainTopologyConfig = field(default_factory=ChainTopologyConfig)


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    links: LinksConfig = field(default_factory=LinksConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        _check_choice("links.rate_model", self.links.rate_model, RATE_MODELS)
        _check_choice("traffic.pair_mode", self.traffic.pair_mode, PAIR_MODES)
        _check_choice("routing.strategy", self.routing.strategy, ROUTING_STRATEGIES)
        _check_choice("topology.type", self.topology.type, TOPOLOGY_TYPES)

        sim = self.simulation
        _check_positive("simulation.duration_s", sim.duration_s)
        _check_positive("simulation.sample_interval_s", sim.sample_interval_s)
        if not 0 <= sim.warmup_s < sim.duration_s:
            raise ConfigError(
                f"simulation.warmup_s must be in [0, duration_s); got {sim.warmup_s}"
            )
        if not sim.seeds:
            raise ConfigError("simulation.seeds must contain at least one seed")

        links = self.links
        _check_positive("links.fixed_rate_bps", links.fixed_rate_bps)
        _check_positive("links.distance.ref_rate_bps", links.distance.ref_rate_bps)
        _check_positive("links.distance.ref_length_km", links.distance.ref_length_km)
        _check_positive("links.pool_capacity_bits", links.pool_capacity_bits)
        if links.distance.attenuation_db_per_km < 0:
            raise ConfigError("links.distance.attenuation_db_per_km must be >= 0")
        if not 0.0 <= links.initial_fill_fraction <= 1.0:
            raise ConfigError("links.initial_fill_fraction must be in [0, 1]")

        traffic = self.traffic
        _check_positive("traffic.key_size_bits", traffic.key_size_bits)
        has_rate = traffic.arrival_rate_per_s is not None
        has_load = traffic.offered_load is not None
        if has_rate == has_load:
            raise ConfigError(
                "specify exactly one of traffic.arrival_rate_per_s and traffic.offered_load "
                "(set the other to null)"
            )
        if has_rate:
            _check_positive("traffic.arrival_rate_per_s", traffic.arrival_rate_per_s)
        if has_load:
            _check_positive("traffic.offered_load", traffic.offered_load)
        if traffic.pair_mode == "fixed":
            pair = traffic.fixed_pair
            if pair is None or len(pair) != 2 or pair[0] == pair[1]:
                raise ConfigError(
                    "traffic.pair_mode 'fixed' requires traffic.fixed_pair: [src, dst] with src != dst"
                )

        relay = self.relay
        _check_positive("relay.timeout_s", relay.timeout_s)
        _check_positive("relay.min_retry_interval_s", relay.min_retry_interval_s)
        if relay.per_hop_processing_s < 0 or relay.propagation_s_per_km < 0:
            raise ConfigError("relay delays must be >= 0")

        _check_positive("routing.kar_epsilon_bits", self.routing.kar_epsilon_bits)

        topo = self.topology
        if topo.n_nodes < 2:
            raise ConfigError("topology.n_nodes must be >= 2")
        _check_positive("topology.area_km", topo.area_km)
        _check_positive("topology.target_avg_degree", topo.target_avg_degree)
        _check_positive("topology.chain.spacing_km", topo.chain.spacing_km)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ConfigError(f"{name} must be one of {list(choices)}; got {value!r}")


def _check_positive(name: str, value: float) -> None:
    if not value > 0:
        raise ConfigError(f"{name} must be > 0; got {value!r}")


# --------------------------------------------------------------------------- dict <-> dataclass


def _from_dict(cls: type, data: Mapping[str, Any], prefix: str = "") -> Any:
    """Build dataclass `cls` from `data`, recursing into nested dataclass fields.

    Missing keys take the dataclass default; unknown keys raise ConfigError.
    """
    if not isinstance(data, Mapping):
        raise ConfigError(f"{prefix or 'config'} must be a mapping; got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        where = prefix.rstrip(".") or "top level"
        raise ConfigError(f"unknown config key(s) at {where}: {unknown}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        nested_cls = _nested_dataclass(cls, name)
        if nested_cls is not None:
            kwargs[name] = _from_dict(nested_cls, value, prefix=f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _nested_dataclass(cls: type, field_name: str) -> type | None:
    """Return the dataclass type of a nested section field, or None for plain values."""
    default = cls()  # every config dataclass is default-constructible
    value = getattr(default, field_name)
    return type(value) if dataclasses.is_dataclass(value) else None


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: `override` merged onto `base`, recursing into nested mappings.

    Non-mapping values (including lists) in `override` replace those in `base`.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_from_dict(data: Mapping[str, Any] | None) -> Config:
    """Build and validate a Config from a (possibly partial) nested dict."""
    cfg = _from_dict(Config, data or {})
    cfg.validate()
    return cfg


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of a config file must be a mapping")
    return data


def load_config(*paths: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load config files in order (later files win), then apply `overrides`.

    With no paths, returns the dataclass defaults. Keys missing from every file fall
    back to the dataclass defaults.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(path))
    if overrides:
        merged = deep_merge(merged, overrides)
    return config_from_dict(merged)
