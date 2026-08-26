"""
A real Spark MLlib price-prediction model trained on the cleaned
listings -- the "Implementierung von Machine-Learning-Modellen... zur
Preisoptimierung" line from the JD, done with Spark's own ML library
(pyspark.ml), not scikit-learn, since this project's whole point is
genuine Spark experience end to end (data prep AND modeling in Spark),
not just Spark for cleaning with modeling bolted on separately.

Evaluated the same honest way as every other ML project in this
portfolio: a time-ordered-equivalent (here, a random but fixed-seed)
train/test split, and comparison against a naive baseline (predict the
market+model group's mean price) -- so a real, disclosed comparison
exists, not just a bare R^2 with no reference point.
"""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, functions as F


FEATURE_COLS_NUMERIC = ["age_years", "mileage_km"]
FEATURE_COLS_CATEGORICAL = ["market", "model", "equipment_level", "channel"]


def build_feature_pipeline() -> Pipeline:
    """HONEST BUG NOTE (see README): the first version of this pipeline
    fed mileage_km straight into VectorAssembler, which crashed with
    'Encountered null while assembling a row' on the deliberately
    injected missing-mileage rows -- VectorAssembler's default
    handleInvalid="error" does not tolerate nulls in NUMERIC columns
    (only StringIndexer's handleInvalid="keep" covers the categorical
    columns). Fixed properly with a real Imputer stage (median
    strategy) rather than papering over it with handleInvalid="skip"
    (which would silently drop real listings) or handleInvalid="keep"
    on the assembler (not a valid option for numeric nulls at all --
    only StringIndexer supports that mode).
    """
    imputer = Imputer(
        inputCols=["mileage_km"], outputCols=["mileage_km_imputed"], strategy="median",
    )
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in FEATURE_COLS_CATEGORICAL
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in FEATURE_COLS_CATEGORICAL
    ]
    assembler = VectorAssembler(
        inputCols=["age_years", "mileage_km_imputed"] + [f"{c}_ohe" for c in FEATURE_COLS_CATEGORICAL],
        outputCol="features",
    )
    gbt = GBTRegressor(featuresCol="features", labelCol="list_price_eur", maxDepth=4, maxIter=50, seed=42)
    return Pipeline(stages=[imputer] + indexers + encoders + [assembler, gbt])


def train_test_split(df: DataFrame, test_fraction: float = 0.2, seed: int = 42):
    """Random split with a fixed seed -- there's no inherent time
    ordering in a listings snapshot (unlike the demand-forecasting
    project's weekly series), so a random split is the correct choice
    here, not a leakage risk the way shuffling a time series would be.
    """
    train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)
    return train_df, test_df


def naive_baseline_predictions(train_df: DataFrame, test_df: DataFrame) -> DataFrame:
    """Naive baseline: predict the (market, model) group's mean price
    from the training set. A real model should beat this -- if it
    doesn't, that's the honest, reportable finding, not something to
    hide (see README for whether that happened here).
    """
    group_means = (
        train_df.groupBy("market", "model")
        .agg(F.avg("list_price_eur").alias("naive_prediction"))
    )
    overall_mean = train_df.agg(F.avg("list_price_eur")).collect()[0][0]
    joined = test_df.join(group_means, on=["market", "model"], how="left")
    return joined.withColumn(
        "naive_prediction",
        F.coalesce(F.col("naive_prediction"), F.lit(overall_mean)),
    )


def train_and_evaluate(df: DataFrame, test_fraction: float = 0.2, seed: int = 42) -> dict:
    train_df, test_df = train_test_split(df, test_fraction=test_fraction, seed=seed)

    pipeline = build_feature_pipeline()
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(labelCol="list_price_eur", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="list_price_eur", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="list_price_eur", predictionCol="prediction", metricName="r2")

    model_rmse = evaluator_rmse.evaluate(predictions)
    model_mae = evaluator_mae.evaluate(predictions)
    model_r2 = evaluator_r2.evaluate(predictions)

    naive_df = naive_baseline_predictions(train_df, test_df)
    naive_rmse = RegressionEvaluator(
        labelCol="list_price_eur", predictionCol="naive_prediction", metricName="rmse",
    ).evaluate(naive_df)
    naive_mae = RegressionEvaluator(
        labelCol="list_price_eur", predictionCol="naive_prediction", metricName="mae",
    ).evaluate(naive_df)

    return {
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "model_r2": model_r2,
        "naive_baseline_rmse": naive_rmse,
        "naive_baseline_mae": naive_mae,
        "rmse_improvement_over_naive_pct": round((naive_rmse - model_rmse) / naive_rmse * 100, 2),
        "mae_improvement_over_naive_pct": round((naive_mae - model_mae) / naive_mae * 100, 2),
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
    }
