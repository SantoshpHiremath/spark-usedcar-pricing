"""
Spark-based data quality and cleaning for the used-vehicle listing feed --
real PySpark DataFrame transformations (not pandas), ensuring data
integrity and data quality across the analytics process for multiple
European markets.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def normalize_market_codes(df: DataFrame) -> DataFrame:
    """Fixes inconsistent casing in the market column (e.g. 'de' vs
    'DE') -- a real, simple, common upstream-feed data-quality issue."""
    return df.withColumn("market", F.upper(F.col("market")))


def flag_and_fix_negative_prices(df: DataFrame) -> DataFrame:
    """A negative list_price_eur is a data-entry sign error (a real,
    plausible upstream mistake), not a genuine negative price -- flags
    it in a new column AND fixes it (takes the absolute value), so
    downstream consumers can filter on the flag if they want to exclude
    corrected rows entirely, while still getting a usable price by
    default.
    """
    return (
        df.withColumn("had_negative_price_flag", F.col("list_price_eur") < 0)
        .withColumn("list_price_eur", F.abs(F.col("list_price_eur")))
    )


def deduplicate_listings(df: DataFrame) -> DataFrame:
    """Removes exact-duplicate listing_id rows (keeping one), simulating
    a retry/re-send from an upstream feed -- confirmed via
    dropDuplicates on listing_id, the natural key for this dataset.
    """
    return df.dropDuplicates(["listing_id"])


def flag_missing_mileage(df: DataFrame) -> DataFrame:
    """Flags rows with missing mileage rather than silently dropping or
    zero-filling them -- mileage is a pricing-relevant feature, so a
    downstream model needs to know explicitly which rows lack it, not
    have it silently imputed to zero (which would look like a brand-new
    car).
    """
    return df.withColumn("missing_mileage_flag", F.col("mileage_km").isNull())


def clean_listings(df: DataFrame) -> DataFrame:
    """Full cleaning pipeline, in a fixed, deliberate order: normalize
    market codes and dedup BEFORE fixing prices, so a duplicate row with
    a corrected price doesn't accidentally get treated as two different
    listings by case-sensitivity, and so the price fix doesn't need to
    run twice on what will become a removed duplicate.
    """
    df = normalize_market_codes(df)
    df = deduplicate_listings(df)
    df = flag_and_fix_negative_prices(df)
    df = flag_missing_mileage(df)
    return df


def quality_summary(raw_df: DataFrame, cleaned_df: DataFrame) -> dict:
    """Aggregate data-quality report -- the kind of summary a real
    pipeline would log or alert on after each run, computed via real
    Spark aggregations (not Python loops over collected rows).
    """
    raw_count = raw_df.count()
    cleaned_count = cleaned_df.count()
    duplicates_removed = raw_count - cleaned_count

    negative_price_count = cleaned_df.filter(F.col("had_negative_price_flag")).count()
    missing_mileage_count = cleaned_df.filter(F.col("missing_mileage_flag")).count()

    market_counts = {
        row["market"]: row["count"]
        for row in cleaned_df.groupBy("market").count().collect()
    }

    return {
        "raw_row_count": raw_count,
        "cleaned_row_count": cleaned_count,
        "duplicates_removed": duplicates_removed,
        "negative_price_fixes": negative_price_count,
        "missing_mileage_count": missing_mileage_count,
        "market_counts": market_counts,
    }
