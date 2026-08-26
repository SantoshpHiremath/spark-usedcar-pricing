"""
Tests for src/generate_data.py -- confirms the synthetic generator is
deterministic under a fixed seed, covers all configured markets, and
actually contains the deliberately injected data-quality problems that
src/data_quality.py is supposed to catch (if these assertions ever fail,
either the generator changed or the injection rates drifted -- either way
the data-quality tests downstream would silently be testing against
nothing).
"""
from __future__ import annotations

from src.generate_data import generate_listings, MARKETS, MODELS


def test_deterministic_under_fixed_seed():
    rows_a = generate_listings(seed=42)
    rows_b = generate_listings(seed=42)
    assert rows_a == rows_b


def test_different_seed_gives_different_data():
    rows_a = generate_listings(seed=42)
    rows_c = generate_listings(seed=99)
    assert rows_a != rows_c


def test_covers_all_configured_markets():
    rows = generate_listings(seed=42)
    seen_markets = {r["market"].upper() for r in rows}
    assert seen_markets == set(MARKETS)


def test_covers_all_configured_models():
    rows = generate_listings(seed=42)
    seen_models = {r["model"] for r in rows}
    assert seen_models == set(MODELS)


def test_contains_duplicate_listing_ids():
    rows = generate_listings(seed=42)
    ids = [r["listing_id"] for r in rows]
    assert len(ids) != len(set(ids)), "expected some duplicate listing_id rows to be injected"


def test_contains_null_mileage():
    rows = generate_listings(seed=42)
    assert any(r["mileage_km"] is None for r in rows)


def test_contains_negative_prices():
    rows = generate_listings(seed=42)
    assert any(r["list_price_eur"] < 0 for r in rows)


def test_contains_lowercase_market_codes():
    rows = generate_listings(seed=42)
    assert any(r["market"] != r["market"].upper() for r in rows)


def test_all_rows_have_expected_fields():
    rows = generate_listings(seed=42)
    expected_fields = {
        "listing_id", "market", "model", "age_years", "mileage_km",
        "equipment_level", "list_price_eur", "channel",
    }
    for r in rows[:50]:
        assert set(r.keys()) == expected_fields


def test_channel_values_are_valid():
    rows = generate_listings(seed=42)
    channels = {r["channel"] for r in rows}
    assert channels <= {"retail", "wholesale"}


def test_row_count_matches_configured_listing_counts():
    from src.generate_data import MARKET_LISTING_COUNTS
    rows = generate_listings(seed=42)
    # actual row count >= configured sum, since duplicates add extra rows
    assert len(rows) >= sum(MARKET_LISTING_COUNTS.values())
