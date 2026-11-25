"""QKD link key pools and link-rate models.

Assumption A1: each link holds a key pool (bits) that refills continuously at rate_bps up
to capacity_bits. Refill is computed lazily when the link is accessed:
    pool(now) = min(capacity, anchor_bits + rate * (now - anchor_s))
There are no per-bit events.

Reads are side-effect free: only consume() moves the anchor. So the pool level at any
time depends only on the consume history, not on who looked at the link and when (the
sampler, a failed availability check, or a key-aware router). Otherwise each read would
re-round the float state, and runs that differ only in their reads (e.g. SPR vs KAR on a
chain) would drift apart by ulps and break ties differently.

Rounding slack: waiting exactly time_until() seconds can leave the computed pool a
float rounding error (~1e-11 bits) short of the target. A pool within EPS_BITS of the
request counts as sufficient. consume() credits that shortfall as generated key (it is
key the continuous-time model has produced), so the pool never goes negative and
pool == initial + generated - consumed still holds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qkdsim.config import ConfigError, LinksConfig

if TYPE_CHECKING:
    import networkx as nx


EPS_BITS = 1e-6  # rounding slack for availability checks (see module docstring)


class InsufficientKeyError(RuntimeError):
    """Raised when consume() asks for more key than the pool holds."""


class KeyLink:
    """Key pool of one QKD link (assumption A1).

    Counters:
        total_generated_bits: bits that actually entered the pool. Key produced while the
            pool is full is discarded and not counted, so at any time
            pool == initial_bits + total_generated_bits - total_consumed_bits.
        total_consumed_bits: bits removed by consume().
        total_discarded_bits: key generated while the pool was full (lost). Every
            generated bit is either kept or discarded, so
            total_generated_bits + total_discarded_bits == rate_bps * elapsed time
            (up to the EPS_BITS rounding credits described in the module docstring).
    Generated/discarded are reported up to the latest time the link was accessed.
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

        # Pool state: the level was anchor_bits at anchor_s; only consume() moves the anchor.
        self._anchor_s = float(start_time_s)
        self._anchor_bits = float(initial_bits)
        self._last_seen_s = float(start_time_s)  # latest access time (monotonicity check)
        # Generated/discarded bits committed up to the anchor.
        self._generated_to_anchor_bits = 0.0
        self._discarded_to_anchor_bits = 0.0
        self.total_consumed_bits = 0.0
        # Number of consume() calls: the pool state changes only on consume, so observers
        # (KAR's path cache) can use it as a state version.
        self.consume_count = 0

    def _pool_at(self, now: float) -> float:
        """Pool level at `now`. Does not change the pool state (only the access clock)."""
        if now < self._last_seen_s:
            raise ValueError(f"time went backwards: now={now} < last access={self._last_seen_s}")
        self._last_seen_s = now
        pool_bits = self._anchor_bits + self.rate_bps * (now - self._anchor_s)
        return pool_bits if pool_bits < self.capacity_bits else self.capacity_bits

    def _since_anchor(self, t: float) -> tuple[float, float]:
        """(kept_bits, discarded_bits) produced between the anchor and time t."""
        produced_bits = self.rate_bps * (t - self._anchor_s)
        kept_bits = min(self.capacity_bits - self._anchor_bits, produced_bits)
        return kept_bits, produced_bits - kept_bits

    @property
    def total_generated_bits(self) -> float:
        return self._generated_to_anchor_bits + self._since_anchor(self._last_seen_s)[0]

    @property
    def total_discarded_bits(self) -> float:
        return self._discarded_to_anchor_bits + self._since_anchor(self._last_seen_s)[1]

    def available(self, now: float) -> float:
        """Bits in the pool at time `now` (lazy refill, A1)."""
        return self._pool_at(now)

    def peek(self, now: float) -> float:
        """Pool level at `now` without touching ANY state (not even the access clock).

        For observers such as routers. Undefined (raises) for times before the anchor.
        """
        if now < self._anchor_s:
            raise ValueError(f"cannot peek before the last consume: now={now} < {self._anchor_s}")
        pool_bits = self._anchor_bits + self.rate_bps * (now - self._anchor_s)
        return pool_bits if pool_bits < self.capacity_bits else self.capacity_bits

    def consume(self, now: float, bits: float) -> None:
        """Remove `bits` from the pool. Raises InsufficientKeyError if the pool is short
        (by more than the EPS_BITS rounding slack)."""
        if bits < 0:
            raise ValueError(f"bits must be >= 0; got {bits}")
        pool_bits = self._pool_at(now)
        if pool_bits < bits - EPS_BITS:
            raise InsufficientKeyError(f"need {bits} bits, pool holds {pool_bits} at t={now}")
        kept_bits, discarded_bits = self._since_anchor(now)
        rounding_credit_bits = bits - pool_bits if pool_bits < bits else 0.0
        self._generated_to_anchor_bits += kept_bits + rounding_credit_bits
        self._discarded_to_anchor_bits += discarded_bits
        self._anchor_s = now
        self._anchor_bits = pool_bits + rounding_credit_bits - bits  # >= 0
        self.total_consumed_bits += bits
        self.consume_count += 1

    def time_until(self, now: float, bits: float) -> float:
        """Seconds from `now` until the pool holds >= `bits`, ignoring other consumers.

        Returns 0 if already available (within EPS_BITS) and inf if `bits` exceeds
        capacity (or the rate is 0). time_until(now, b) == 0 exactly when consume(now, b)
        would succeed.
        """
        if bits > self.capacity_bits:
            return math.inf
        deficit_bits = bits - self._pool_at(now)
        if deficit_bits <= EPS_BITS:
            return 0.0
        if self.rate_bps == 0:
            return math.inf
        return deficit_bits / self.rate_bps

    def __repr__(self) -> str:
        return (
            f"KeyLink(rate_bps={self.rate_bps}, capacity_bits={self.capacity_bits}, "
            f"anchor_bits={self._anchor_bits} at t={self._anchor_s}, "
            f"length_km={self.length_km})"
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
