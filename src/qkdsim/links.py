"""QKD link key pools and link-rate models.

Assumption A1: each link holds a key pool (bits) that refills continuously at rate_bps up
to capacity_bits. Refill is computed lazily when the link is accessed:
    pool = min(capacity, pool + rate * (now - last_update))
There are no per-bit events.

All comparisons are exact (no epsilon). Because time_until() is computed in floating
point, waiting exactly time_until() seconds can leave the pool a rounding error short;
the simulator's retry loop (min_retry_interval_s) absorbs this.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qkdsim.config import ConfigError, LinksConfig

if TYPE_CHECKING:
    import networkx as nx


class InsufficientKeyError(RuntimeError):
    """Raised when consume() asks for more key than the pool holds."""


class KeyLink:
    """Key pool of one QKD link (assumption A1).

    Counters:
        total_generated_bits: bits that actually entered the pool. Key produced while the
            pool is full is discarded and not counted, so at any time
            pool == initial_bits + total_generated_bits - total_consumed_bits.
        total_consumed_bits: bits removed by consume().
    """

    def __init__(
        self,
        rate_bps: float,
        capacity_bits: float,
        initial_bits: float,
        length_km: float,
        start_time_s: float = 0.0,
    ) -> None:
        if rate_bps < 0:
            raise ValueError(f"rate_bps must be >= 0; got {rate_bps}")
        if capacity_bits <= 0:
            raise ValueError(f"capacity_bits must be > 0; got {capacity_bits}")
        if not 0 <= initial_bits <= capacity_bits:
            raise ValueError(
                f"initial_bits must be in [0, capacity_bits={capacity_bits}]; got {initial_bits}"
            )
        if length_km < 0:
            raise ValueError(f"length_km must be >= 0; got {length_km}")

        self.rate_bps = float(rate_bps)
        self.capacity_bits = float(capacity_bits)
        self.initial_bits = float(initial_bits)
        self.length_km = float(length_km)

        self._pool_bits = float(initial_bits)
        self._last_update_s = float(start_time_s)
        self.total_generated_bits = 0.0
        self.total_consumed_bits = 0.0

    def _refill(self, now: float) -> None:
        if now < self._last_update_s:
            raise ValueError(
                f"time went backwards: now={now} < last_update={self._last_update_s}"
            )
        added_bits = min(
            self.capacity_bits - self._pool_bits,
            self.rate_bps * (now - self._last_update_s),
        )
        self._pool_bits += added_bits
        self.total_generated_bits += added_bits
        self._last_update_s = now

    def available(self, now: float) -> float:
        """Bits in the pool at time `now` (after lazy refill)."""
        self._refill(now)
        return self._pool_bits

    def consume(self, now: float, bits: float) -> None:
        """Remove `bits` from the pool. Raises InsufficientKeyError if the pool is short."""
        if bits < 0:
            raise ValueError(f"bits must be >= 0; got {bits}")
        self._refill(now)
        if bits > self._pool_bits:
            raise InsufficientKeyError(
                f"need {bits} bits, pool holds {self._pool_bits} at t={now}"
            )
        self._pool_bits -= bits
        self.total_consumed_bits += bits

    def time_until(self, now: float, bits: float) -> float:
        """Seconds from `now` until the pool holds >= `bits`, ignoring other consumers.

        Returns 0 if already available and inf if `bits` exceeds capacity (or the rate is 0).
        """
        if bits > self.capacity_bits:
            return math.inf
        self._refill(now)
        deficit_bits = bits - self._pool_bits
        if deficit_bits <= 0:
            return 0.0
        if self.rate_bps == 0:
            return math.inf
        return deficit_bits / self.rate_bps

    def __repr__(self) -> str:
        return (
            f"KeyLink(rate_bps={self.rate_bps}, capacity_bits={self.capacity_bits}, "
            f"pool_bits={self._pool_bits}, length_km={self.length_km}, "
            f"last_update_s={self._last_update_s})"
        )


def link_rate(length_km: float, cfg: LinksConfig, skr_bps: float | None = None) -> float:
    """Secret-key rate of a link (bits/s) under cfg.rate_model.

    fixed:    cfg.fixed_rate_bps for every link.
    distance: R(L) = ref_rate_bps * 10^(-alpha * (L - ref_length_km) / 10).
              Simplification: real secret-key-rate curves fall faster near maximum range.
    per_link: `skr_bps`, taken from the topology data file (e.g. data/secoqc.yaml).
    """
    if cfg.rate_model == "fixed":
        return cfg.fixed_rate_bps
    if cfg.rate_model == "distance":
        d = cfg.distance
        return d.ref_rate_bps * 10 ** (-d.attenuation_db_per_km * (length_km - d.ref_length_km) / 10)
    if cfg.rate_model == "per_link":
        if skr_bps is None:
            raise ConfigError(
                "rate_model 'per_link' needs a per-link rate (skr_bps) from the topology "
                "data file, but none was given for this link"
            )
        return float(skr_bps)
    raise ConfigError(f"unknown rate_model {cfg.rate_model!r}")


def attach_links(G: nx.Graph, cfg: LinksConfig) -> nx.Graph:
    """Add a KeyLink to every edge of G as edge attribute "link" (modifies G in place).

    Rates come from link_rate(); for rate_model 'per_link' each edge must carry an
    "skr_bps" attribute (only the SECOQC topology does). Every link starts with
    initial_fill_fraction * pool_capacity_bits.
    """
    if cfg.rate_model == "per_link":
        missing = [(u, v) for u, v, skr in G.edges(data="skr_bps") if skr is None]
        if missing:
            shown = ", ".join(f"{u}-{v}" for u, v in missing[:5])
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise ConfigError(
                "rate_model 'per_link' requires per-link rates (edge attribute skr_bps) from "
                f"the topology data file, but {len(missing)} of {G.number_of_edges()} edges "
                f"have none: {shown}{more}. Use rate_model 'fixed' or 'distance' for this "
                "topology."
            )

    initial_bits = cfg.initial_fill_fraction * cfg.pool_capacity_bits
    for _, _, data in G.edges(data=True):
        rate_bps = link_rate(data["length_km"], cfg, data.get("skr_bps"))
        data["link"] = KeyLink(
            rate_bps=rate_bps,
            capacity_bits=cfg.pool_capacity_bits,
            initial_bits=initial_bits,
            length_km=data["length_km"],
        )
    return G
