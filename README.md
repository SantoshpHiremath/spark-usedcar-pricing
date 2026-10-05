# spark-usedcar-pricing

An Apache Spark (PySpark) data pipeline for used-vehicle pricing analytics across multiple European markets: synthetic data generation, Spark DataFrame-based data-quality cleaning, pricing analytics via window functions and aggregations, and a Spark MLlib price-prediction model evaluated against a naive baseline.

## What it does

The stack is SQL-style analytics, Python, and Apache Spark, covering cross-market data quality, pricing analytics, and ML-based price optimization.

- Generates synthetic multi-market listings with injected data-quality problems.
- Cleans them with Spark DataFrame transformations (deduplication, price-sign fixes, market-code normalization, missing-mileage flagging).
- Analyzes prices with window functions, aggregations, and a cross-market price-gap view.
- Trains a Spark MLlib `GBTRegressor` pipeline and compares it with a naive group-mean baseline.

Every module runs on an actual local Spark session (`pyspark.sql.SparkSession`, `local[*]` master), using genuine DataFrame transformations, window functions (`pyspark.sql.Window`), and Spark MLlib (`pyspark.ml`).

## Data

- **The data is synthetic.** `src/generate_data.py` generates ~2,900 listings across 5 markets (DE/FR/IT/ES/NL) and 6 model lines, with a depreciation-plus-mileage pricing formula and per-market price multipliers, so the pricing signal is internally consistent and verifiable against the injected `MARKET_PRICE_MULTIPLIER` config. The pipeline is built so real listing feeds can replace it.
- **Data-quality problems are deliberately injected** (duplicate listing IDs, missing mileage, negative-price sign errors, inconsistent market-code casing) so the cleaning pipeline has real, verifiable problems to catch and fix.

## Scope

Spark runs in local mode. AWS Glue is a managed Spark job runtime with its own job-authoring conventions (Glue jobs, Glue Data Catalog, the `awsglue` library, DynamicFrames); the DataFrame API, window functions, and ML pipelines here are standard Spark and carry over, with the Glue-specific job wrapper added at deployment.

## Results

Run via `python run_pipeline.py` (full output also captured in the test suite):

- Raw listings: 2,899 rows, cleaned: 2,850 rows (49 duplicates removed, 24 negative-price sign errors fixed, 60 missing-mileage rows flagged).
- Cross-market price gap vs. DE reference, X5 model: NL +7.6%, FR +3.73%, IT -4.9%, ES -7.34%. These line up directionally and in rough magnitude with the injected `MARKET_PRICE_MULTIPLIER` config (NL=1.10, ES=0.92 vs DE=1.00), confirming the window-function/join-based cross-market analytics compute the right thing.
- GBTRegressor price model vs. naive (market+model group mean) baseline:
  - Model RMSE ~1,235 EUR vs. naive RMSE ~5,920 EUR (**~79% improvement**)
  - Model MAE ~916 EUR vs. naive MAE ~4,875 EUR (**~81% improvement**)
  - Model R² ~0.984

The strong result reflects the synthetic price formula's deterministic structure (a clean depreciation curve plus a small noise term); a real market would carry more unexplained variance. The project's focus is a correctly built Spark ML pipeline and a sound baseline-comparison methodology.

## Tests

`pytest -q`: 32 tests, run against a real local Spark session (see `conftest.py`).

## Project structure

```
src/
  spark_session.py     -- local SparkSession factory
  generate_data.py     -- synthetic multi-market listing generator
  data_quality.py      -- Spark-based cleaning (dedup, price-sign fix,
                           market-code normalization, missing-mileage
                           flagging) + an aggregate quality_summary()
  pricing_analytics.py -- window-function price ranking within
                           market+model groups, per-market/model
                           aggregation, cross-market price-gap analysis
  price_model.py        -- Spark MLlib GBTRegressor pipeline (Imputer +
                           StringIndexer + OneHotEncoder + VectorAssembler
                           + GBTRegressor), evaluated against a naive
                           group-mean baseline
tests/                  -- pytest suite, 32 tests, run against a real
                           local Spark session (see conftest.py)
run_pipeline.py          -- end-to-end demo script
```

## Running it

```
pip install -r requirements.txt
python run_pipeline.py     # end-to-end demo
pytest -q                  # full test suite (32 tests, real local Spark)
```

Requires a local Java installation (Java 8/11/17/21 all work with the pinned PySpark version), since Spark runs on the JVM even when driven from Python.

## Notes

The first version of `price_model.build_feature_pipeline()` fed `mileage_km` directly into `VectorAssembler` alongside the categorical columns. It crashed with:

```
org.apache.spark.SparkException: Encountered null while assembling a row
with handleInvalid = "error". Consider removing nulls from dataset or
using handleInvalid = "keep" or "skip".
```

`VectorAssembler`'s default `handleInvalid="error"` does not tolerate nulls in **numeric** input columns. The `StringIndexer(handleInvalid="keep")` stages already used for the categorical columns only cover the string columns they index, not a separate numeric column feeding the same assembler. The deliberately injected null `mileage_km` rows from `generate_data.py` (~2% of rows) surfaced this in testing.

**Fix**: I added a `pyspark.ml.feature.Imputer(strategy="median")` stage before the assembler, producing `mileage_km_imputed`, which is what feeds `VectorAssembler`. I chose imputation over `VectorAssembler(handleInvalid="skip")`, which would have silently dropped real listings with missing mileage from training and prediction, a worse outcome for a pricing model. `tests/test_price_model.py::test_feature_pipeline_handles_null_mileage_without_crashing` is a regression test guarding against this bug being reintroduced.

## Possible extensions

- Wrap the pipeline as an AWS Glue job (Glue Data Catalog, DynamicFrames).
- Run against a real multi-market listing feed.
- Add a dashboard layer on top of the pricing analytics.
