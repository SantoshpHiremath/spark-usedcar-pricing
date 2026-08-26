"""
End-to-end demo: generate synthetic multi-market listing data -> clean it
with real Spark data-quality transforms -> run pricing analytics
(per-market ranking, aggregation, cross-market gap) -> train and evaluate
a real Spark MLlib price-prediction model against a naive baseline ->
print a summary of everything.

Run with: python run_pipeline.py
"""
from __future__ import annotations

import json

from src.spark_session import get_spark
from src.generate_data import generate_listings
from src.data_quality import clean_listings, quality_summary
from src.pricing_analytics import market_price_summary, cross_market_price_gap
from src.price_model import train_and_evaluate


def main():
    spark = get_spark(app_name="used_car_pricing_pipeline_demo")

    print("=== 1. Generating synthetic multi-market listing data ===")
    rows = generate_listings(seed=42)
    raw_df = spark.createDataFrame(rows)
    print(f"Generated {raw_df.count()} raw listing rows.\n")

    print("=== 2. Cleaning with real Spark data-quality transforms ===")
    cleaned_df = clean_listings(raw_df).cache()
    summary = quality_summary(raw_df, cleaned_df)
    print(json.dumps(summary, indent=2))
    print()

    print("=== 3. Per-market, per-model price summary (first 10 rows) ===")
    market_price_summary(cleaned_df).show(10, truncate=False)

    print("=== 4. Cross-market price gap vs. DE (X5 model) ===")
    gaps = cross_market_price_gap(cleaned_df, reference_market="DE")
    gaps.filter(gaps.model == "X5").orderBy("market").show(truncate=False)

    print("=== 5. Training and evaluating GBTRegressor price model ===")
    results = train_and_evaluate(cleaned_df, test_fraction=0.2, seed=42)
    print(json.dumps(results, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
