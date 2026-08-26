"""
Tests for src/data_quality.py -- each cleaning function individually,
then the end-to-end clean_listings() pipeline and quality_summary()
against real generated data, run through a real (local) Spark session
(the `spark` fixture in conftest.py), not mocked.
"""
from __future__ import annotations

from src.generate_data import generate_listings
from src.data_quality import (
    normalize_market_codes,
    flag_and_fix_negative_prices,
    deduplicate_listings,
    flag_missing_mileage,
    clean_listings,
    quality_summary,
)


def _raw_df(spark):
    rows = generate_listings(seed=42)
    return spark.createDataFrame(rows)


def test_normalize_market_codes_uppercases_all(spark):
    df = _raw_df(spark)
    result = normalize_market_codes(df)
    markets = [r["market"] for r in result.select("market").distinct().collect()]
    assert all(m == m.upper() for m in markets)


def test_flag_and_fix_negative_prices(spark):
    df = _raw_df(spark)
    result = flag_and_fix_negative_prices(df)
    flagged = result.filter(result.had_negative_price_flag)
    assert flagged.count() > 0
    # every flagged row should now have a non-negative price
    assert flagged.filter(flagged.list_price_eur < 0).count() == 0
    # no row should have a negative price left anywhere
    assert result.filter(result.list_price_eur < 0).count() == 0


def test_deduplicate_listings_removes_duplicate_ids(spark):
    df = _raw_df(spark)
    raw_count = df.count()
    deduped = deduplicate_listings(df)
    deduped_count = deduped.count()
    assert deduped_count < raw_count
    # no duplicate listing_ids remain
    distinct_ids = deduped.select("listing_id").distinct().count()
    assert distinct_ids == deduped_count


def test_flag_missing_mileage(spark):
    df = _raw_df(spark)
    result = flag_missing_mileage(df)
    flagged = result.filter(result.missing_mileage_flag)
    assert flagged.count() > 0
    assert flagged.filter(flagged.mileage_km.isNotNull()).count() == 0


def test_clean_listings_end_to_end_no_crash(spark):
    df = _raw_df(spark)
    cleaned = clean_listings(df)
    assert cleaned.count() > 0
    cols = set(cleaned.columns)
    assert {"had_negative_price_flag", "missing_mileage_flag"} <= cols


def test_clean_listings_no_negative_prices_remain(spark):
    df = _raw_df(spark)
    cleaned = clean_listings(df)
    assert cleaned.filter(cleaned.list_price_eur < 0).count() == 0


def test_clean_listings_no_duplicate_ids_remain(spark):
    df = _raw_df(spark)
    cleaned = clean_listings(df)
    assert cleaned.select("listing_id").distinct().count() == cleaned.count()


def test_quality_summary_counts_are_internally_consistent(spark):
    raw = _raw_df(spark)
    cleaned = clean_listings(raw)
    summary = quality_summary(raw, cleaned)

    assert summary["raw_row_count"] == raw.count()
    assert summary["cleaned_row_count"] == cleaned.count()
    assert summary["duplicates_removed"] == summary["raw_row_count"] - summary["cleaned_row_count"]
    assert summary["duplicates_removed"] > 0
    assert summary["negative_price_fixes"] > 0
    assert summary["missing_mileage_count"] > 0
    assert set(summary["market_counts"].keys()) == {"DE", "FR", "IT", "ES", "NL"}
    assert sum(summary["market_counts"].values()) == summary["cleaned_row_count"]
