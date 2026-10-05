"""
Tests for src/price_model.py -- confirms the full Spark MLlib pipeline
trains without error (including on data with the deliberately injected
null mileage values, which is a regression test for the VectorAssembler
bug documented in the README and in build_feature_pipeline()'s docstring),
that it beats the naive baseline, and that the returned metrics are in
sane ranges.
"""
from __future__ import annotations

import pytest

from src.generate_data import generate_listings
from src.data_quality import clean_listings
from src.price_model import (
    build_feature_pipeline,
    train_test_split,
    naive_baseline_predictions,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def cleaned_df(spark):
    rows = generate_listings(seed=42)
    raw = spark.createDataFrame(rows)
    return clean_listings(raw).cache()


def test_train_test_split_is_disjoint_and_covers_all_rows(cleaned_df):
    train_df, test_df = train_test_split(cleaned_df, test_fraction=0.2, seed=42)
    train_ids = {r["listing_id"] for r in train_df.select("listing_id").collect()}
    test_ids = {r["listing_id"] for r in test_df.select("listing_id").collect()}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == cleaned_df.count()


def test_naive_baseline_predictions_no_nulls(cleaned_df):
    train_df, test_df = train_test_split(cleaned_df, test_fraction=0.2, seed=42)
    naive_df = naive_baseline_predictions(train_df, test_df)
    assert naive_df.filter(naive_df.naive_prediction.isNull()).count() == 0


def test_feature_pipeline_handles_null_mileage_without_crashing(cleaned_df):
    """Regression test for the real VectorAssembler null-handling bug
    (see README 'Notes' section and build_feature_pipeline()'s
    docstring): the cleaned dataset still contains missing_mileage_flag
    rows with a genuinely null mileage_km, and the pipeline must fit and
    transform them without raising, thanks to the Imputer stage.
    """
    assert cleaned_df.filter(cleaned_df.mileage_km.isNull()).count() > 0, \
        "test fixture should contain null mileage rows to exercise the bug fix"

    train_df, test_df = train_test_split(cleaned_df, test_fraction=0.2, seed=42)
    pipeline = build_feature_pipeline()
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    # should not raise, and every row should get a prediction, including
    # ones with originally-null mileage
    assert predictions.filter(predictions.prediction.isNull()).count() == 0


def test_train_and_evaluate_returns_expected_keys(cleaned_df):
    results = train_and_evaluate(cleaned_df, test_fraction=0.2, seed=42)
    expected_keys = {
        "model_rmse", "model_mae", "model_r2",
        "naive_baseline_rmse", "naive_baseline_mae",
        "rmse_improvement_over_naive_pct", "mae_improvement_over_naive_pct",
        "train_rows", "test_rows",
    }
    assert expected_keys <= set(results.keys())


def test_train_and_evaluate_beats_naive_baseline(cleaned_df):
    results = train_and_evaluate(cleaned_df, test_fraction=0.2, seed=42)
    assert results["model_rmse"] < results["naive_baseline_rmse"]
    assert results["model_mae"] < results["naive_baseline_mae"]
    assert results["rmse_improvement_over_naive_pct"] > 0


def test_train_and_evaluate_metrics_in_sane_ranges(cleaned_df):
    results = train_and_evaluate(cleaned_df, test_fraction=0.2, seed=42)
    assert 0 <= results["model_r2"] <= 1
    assert results["model_rmse"] > 0
    assert results["model_mae"] > 0
    assert results["train_rows"] > 0
    assert results["test_rows"] > 0
