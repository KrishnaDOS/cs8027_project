"""Tests for KeyLink (assumption A1), link_rate and attach_links."""

import math

import pytest

from qkdsim.config import ConfigError, DistanceRateConfig, LinksConfig
from qkdsim.links import InsufficientKeyError, KeyLink, attach_links, link_rate
from qkdsim.topologies import make_chain


def make_link(rate_bps: float = 100.0, capacity_bits: float = 1000.0, initial_bits: float = 200.0) -> KeyLink:
    return KeyLink(rate_bps=rate_bps, capacity_bits=capacity_bits, initial_bits=initial_bits, length_km=10.0)


# --------------------------------------------------------------------------- refill


def test_initial_pool() -> None:
    assert make_link().available(0.0) == 200.0


def test_refill_is_linear_in_time() -> None:
    link = make_link()
    assert link.available(2.0) == pytest.approx(400.0)
    assert link.available(5.0) == pytest.approx(700.0)


def test_capacity_clamp() -> None:
    link = make_link()
    assert link.available(100.0) == 1000.0  # 200 + 100*100 clamped to 1000
    assert link.available(200.0) == 1000.0


def test_refill_continues_after_clamp_and_consume() -> None:
    link = make_link()
    link.available(100.0)  # full
    link.consume(100.0, 600.0)
    assert link.available(101.0) == pytest.approx(500.0)


def test_time_backwards_raises() -> None:
    link = make_link()
    link.available(5.0)
    with pytest.raises(ValueError, match="backwards"):
        link.available(4.0)


# --------------------------------------------------------------------------- consume


def test_consume_debits_pool() -> None:
    link = make_link()
    link.consume(1.0, 250.0)  # pool 300 -> 50
    assert link.available(1.0) == pytest.approx(50.0)


def test_consume_raises_when_insufficient_and_leaves_pool_untouched() -> None:
    link = make_link()
    with pytest.raises(InsufficientKeyError):
        link.consume(0.0, 200.5)
    assert link.available(0.0) == 200.0
    assert link.total_consumed_bits == 0.0


def test_consume_exact_amount_empties_pool() -> None:
    link = make_link()
    link.consume(0.0, 200.0)
    assert link.available(0.0) == 0.0
    with pytest.raises(InsufficientKeyError):
        link.consume(0.0, 1.0)


def test_consume_negative_raises() -> None:
    with pytest.raises(ValueError):
        make_link().consume(0.0, -1.0)


# --------------------------------------------------------------------------- time_until


def test_time_until_zero_when_available() -> None:
    assert make_link().time_until(0.0, 150.0) == 0.0
    assert make_link().time_until(0.0, 200.0) == 0.0


def test_time_until_deficit_over_rate() -> None:
    link = make_link()
    wait_s = link.time_until(1.0, 800.0)  # pool 300 at t=1, deficit 500 at 100 bps
    assert wait_s == pytest.approx(5.0)
    assert link.available(1.0 + wait_s) == pytest.approx(800.0)


def test_time_until_inf_above_capacity() -> None:
    assert make_link().time_until(0.0, 1000.1) == math.inf
    assert make_link().time_until(0.0, 1000.0) == pytest.approx(8.0)


def test_time_until_inf_with_zero_rate() -> None:
    assert make_link(rate_bps=0.0).time_until(0.0, 500.0) == math.inf


# --------------------------------------------------------------------------- counters


def test_counters_and_conservation() -> None:
    link = make_link()
    link.consume(1.0, 250.0)  # generated 100, pool 50
    link.consume(3.0, 200.0)  # generated 200 more, pool 50
    link.available(100.0)  # fills to capacity: generated 950 more (rest discarded)
    assert link.total_consumed_bits == pytest.approx(450.0)
    assert link.total_generated_bits == pytest.approx(1250.0)
    pool = link.available(100.0)
    assert pool == pytest.approx(link.initial_bits + link.total_generated_bits - link.total_consumed_bits)
    assert link.total_consumed_bits <= link.initial_bits + link.total_generated_bits
    assert link.total_discarded_bits == pytest.approx(100 * 100.0 - 1250.0)  # produced - kept


def test_discarded_bits_account_for_all_key() -> None:
    link = make_link()  # 100 bps, capacity 1000, initial 200, start t=0
    link.available(20.0)  # fills at t=8; 1200 bits discarded
    assert link.total_discarded_bits == pytest.approx(1200.0)
    link.consume(20.0, 900.0)  # pool 100
    link.available(25.0)  # +500, nothing discarded
    link.consume(25.0, 600.0)  # pool 0
    link.available(40.0)  # +1000 by t=35, then 500 discarded
    link.available(40.0)  # repeated access at the same time changes nothing
    elapsed_s = 40.0
    assert link.total_discarded_bits == pytest.approx(1700.0)
    assert (
        link.initial_bits + link.total_generated_bits + link.total_discarded_bits
        == pytest.approx(link.rate_bps * elapsed_s + link.initial_bits)
    )


def test_no_discard_below_capacity() -> None:
    link = make_link()
    link.available(5.0)  # pool 700 < 1000
    assert link.total_discarded_bits == 0.0


# --------------------------------------------------------------------------- link_rate


def test_fixed_rate_ignores_length() -> None:
    cfg = LinksConfig(rate_model="fixed", fixed_rate_bps=5000)
    assert link_rate(1.0, cfg) == 5000
    assert link_rate(500.0, cfg) == 5000


def test_distance_model_reference_points() -> None:
    cfg = LinksConfig(rate_model="distance")  # ref 5000 bps at 25 km, 0.2 dB/km
    assert link_rate(25.0, cfg) == pytest.approx(5000.0)
    assert link_rate(75.0, cfg) == pytest.approx(500.0)  # 50 km * 0.2 dB/km = 10 dB


def test_distance_model_custom_attenuation() -> None:
    cfg = LinksConfig(
        rate_model="distance",
        distance=DistanceRateConfig(ref_rate_bps=1000, ref_length_km=10, attenuation_db_per_km=0.25),
    )
    assert link_rate(50.0, cfg) == pytest.approx(100.0)  # 40 km * 0.25 = 10 dB


def test_per_link_rate_needs_skr() -> None:
    cfg = LinksConfig(rate_model="per_link")
    assert link_rate(10.0, cfg, skr_bps=3100) == 3100
    with pytest.raises(ConfigError, match="per_link"):
        link_rate(10.0, cfg)


# --------------------------------------------------------------------------- attach_links


def test_attach_links_fixed_on_chain() -> None:
    cfg = LinksConfig()  # fixed 5000 bps, capacity 200000, fill 0.5
    G = attach_links(make_chain(5, 10.0), cfg)
    for _, _, data in G.edges(data=True):
        link = data["link"]
        assert isinstance(link, KeyLink)
        assert link.rate_bps == 5000
        assert link.capacity_bits == 200_000
        assert link.initial_bits == 100_000
        assert link.length_km == data["length_km"]


def test_attach_links_distance_uses_edge_length() -> None:
    G = attach_links(make_chain(3, 75.0), LinksConfig(rate_model="distance"))
    assert all(d["link"].rate_bps == pytest.approx(500.0) for _, _, d in G.edges(data=True))
