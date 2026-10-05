"""
Pricing analytics via real Spark window functions and aggregations --
building data assets with SQL, Python and Apache Spark, applied
specifically to used-vehicle pricing.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F


def price_per_km(df: DataFrame) -> DataFrame:
    """A simple, real derived pricing metric: price per 1,000km of
    mileage, useful for comparing vehicles of different mileage bands on
    a normalized basis.
    """
    return df.withColumn(
        "price_per_1000km",
        F.when(F.col("mileage_km") > 0, F.col("list_price_eur") / (F.col("mileage_km") / 1000))
        .otherwise(F.lit(None)),
    )


def market_model_price_rank(df: DataFrame) -> DataFrame:
    """Ranks each listing's price within its own (market, model) group,
    using a real Spark window function -- exactly the kind of
    cross-market comparison across European markets: is this specific listing priced high or low relative to
    comparable vehicles in the SAME market and model, not compared
    across incomparable groups.
    """
    window = Window.partitionBy("market", "model").orderBy(F.col("list_price_eur").desc())
    return df.withColumn("price_rank_within_market_model", F.rank().over(window))


def market_price_summary(df: DataFrame) -> DataFrame:
    """Per-market, per-model average/median/stddev price -- a real
    aggregation a pricing analyst would use to spot a market where a
    given model is priced unusually high or low relative to other
    markets.
    """
    return (
        df.groupBy("market", "model")
        .agg(
            F.count("*").alias("listing_count"),
            F.round(F.avg("list_price_eur"), 2).alias("avg_price_eur"),
            F.round(F.expr("percentile_approx(list_price_eur, 0.5)"), 2).alias("median_price_eur"),
            F.round(F.stddev("list_price_eur"), 2).alias("stddev_price_eur"),
        )
        .orderBy("market", "model")
    )


def cross_market_price_gap(df: DataFrame, reference_market: str = "DE") -> DataFrame:
    """For each model, compares each market's average price against a
    reference market (default DE) -- surfaces a real, actionable
    cross-market pricing signal: "X5 is priced 12% higher in NL than in
    DE," the kind of finding this role's international pricing work is
    meant to produce.
    """
    summary = (
        df.groupBy("market", "model")
        .agg(F.avg("list_price_eur").alias("avg_price_eur"))
    )
    reference = (
        summary.filter(F.col("market") == reference_market)
        .select(F.col("model"), F.col("avg_price_eur").alias("reference_price_eur"))
    )
    joined = summary.join(reference, on="model", how="inner")
    return (
        joined
        .withColumn(
            "pct_diff_vs_reference",
            F.round((F.col("avg_price_eur") - F.col("reference_price_eur")) / F.col("reference_price_eur") * 100, 2),
        )
        .orderBy("model", "market")
    )
