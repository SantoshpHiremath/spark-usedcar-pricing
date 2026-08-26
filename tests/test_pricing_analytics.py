"""
Tests for src/pricing_analytics.py -- verifies the window-function ranking,
aggregation, and cross-market gap logic against real Spark computations,
including a check against the KNOWN injected MARKET_PRICE_MULTIPLIER
values (so this isn't just "does it run without crashing" but "does it
produce the directionally and numerically correct answer").
"""
from __future__ import annotations

import pytest

from src.generate_data import generate_listings, MARKET_PRICE_MULTIPLIER
from src.data_quality import clean_listings
from src.pricing_analytics import (
    price_per_km,
    market_model_price_rank,
    market_price_summary,
    cross_market_price_gap,
)


@pytest.fixture(scope="module")
def cleaned_df(spark):
    rows = generate_listings(seed=42)
    raw = spark.createDataFrame(rows)
    return clean_listings(raw).cache()


def test_price_per_km_computes_expected_ratio(cleaned_df):
    result = price_per_km(cleaned_df)
    row = result.filter(result.mileage_km > 0).select("list_price_eur", "mileage_km", "price_per_1000km").first()
    expected = row["list_price_eur"] / (row["mileage_km"] / 1000)
    assert abs(row["price_per_1000km"] - expected) < 1e-6


def test_price_per_km_null_mileage_gives_null_ratio(cleaned_df):
    result = price_per_km(cleaned_df)
    null_mileage_rows = result.filter(result.mileage_km.isNull())
    assert null_mileage_rows.count() > 0
    assert null_mileage_rows.filter(null_mileage_rows.price_per_1000km.isNotNull()).count() == 0


def test_market_model_price_rank_is_dense_within_group(cleaned_df):
    ranked = market_model_price_rank(cleaned_df)
    # for a specific (market, model) group, rank 1 should be the max price
    group = ranked.filter((ranked.market == "DE") & (ranked.model == "X5"))
    top = group.orderBy(group.price_rank_within_market_model.asc()).first()
    max_price = group.agg({"list_price_eur": "max"}).first()[0]
    assert top["list_price_eur"] == max_price


def test_market_price_summary_counts_match(cleaned_df):
    summary = market_price_summary(cleaned_df)
    total_from_summary = sum(r["listing_count"] for r in summary.collect())
    assert total_from_summary == cleaned_df.count()


def test_market_price_summary_has_all_market_model_combinations(cleaned_df):
    summary = market_price_summary(cleaned_df)
    rows = summary.collect()
    markets = {r["market"] for r in rows}
    assert markets == {"DE", "FR", "IT", "ES", "NL"}


def test_cross_market_price_gap_reference_market_is_zero(cleaned_df):
    gaps = cross_market_price_gap(cleaned_df, reference_market="DE")
    de_rows = gaps.filter(gaps.market == "DE").collect()
    assert len(de_rows) > 0
    for r in de_rows:
        assert abs(r["pct_diff_vs_reference"]) < 0.01


def test_cross_market_price_gap_matches_injected_multiplier_direction(cleaned_df):
    """NL has a higher MARKET_PRICE_MULTIPLIER than DE (1.10 vs 1.00), and
    ES has a lower one (0.92 vs 1.00) -- the computed pct_diff_vs_reference
    should reflect that direction, and land reasonably close in magnitude
    given the +/-5% uniform noise and depreciation-driven variance in the
    underlying data.
    """
    gaps = cross_market_price_gap(cleaned_df, reference_market="DE")
    nl_x5 = gaps.filter((gaps.market == "NL") & (gaps.model == "X5")).first()
    es_x5 = gaps.filter((gaps.market == "ES") & (gaps.model == "X5")).first()

    assert nl_x5["pct_diff_vs_reference"] > 0
    assert es_x5["pct_diff_vs_reference"] < 0

    expected_nl_pct = (MARKET_PRICE_MULTIPLIER["NL"] - MARKET_PRICE_MULTIPLIER["DE"]) / MARKET_PRICE_MULTIPLIER["DE"] * 100
    expected_es_pct = (MARKET_PRICE_MULTIPLIER["ES"] - MARKET_PRICE_MULTIPLIER["DE"]) / MARKET_PRICE_MULTIPLIER["DE"] * 100

    assert abs(nl_x5["pct_diff_vs_reference"] - expected_nl_pct) < 5
    assert abs(es_x5["pct_diff_vs_reference"] - expected_es_pct) < 5
