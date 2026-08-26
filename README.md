# spark-usedcar-pricing

A real Apache Spark (PySpark) data pipeline for used-vehicle pricing
analytics across multiple European markets: synthetic data generation,
Spark DataFrame-based data-quality cleaning, pricing analytics via window
functions and aggregations, and a Spark MLlib price-prediction model
evaluated against a naive baseline.

Built specifically to close a gap for a Working Student "Used Car Pricing
Analytics" posting whose core stack is SQL, Python, and **Apache Spark
(AWS Glue)**, cross-market data quality, dashboards, and ML-based price
optimization.

## What this is (and isn't)

- **The data is synthetic**, not real BMW, dealer, or customer data,
  which I have no access to. `src/generate_data.py` generates ~2,900
  listings across 5 markets (DE/FR/IT/ES/NL) and 6 model lines, with a
  depreciation-plus-mileage pricing formula and per-market price
  multipliers, so the pricing signal in the data is real and internally
  consistent (verifiable against the injected `MARKET_PRICE_MULTIPLIER`
  config), even though the underlying listings are fabricated.
- **Data-quality problems are deliberately injected** (duplicate
  listing IDs, missing mileage, negative-price sign errors, inconsistent
  market-code casing) so the cleaning pipeline has real, verifiable
  problems to catch and fix -- not a clean textbook table with nothing to
  do.
- **Spark is real and local**, not mocked: every module runs on an
  actual local Spark session (`pyspark.sql.SparkSession`, `local[*]`
  master), using genuine DataFrame transformations, window functions
  (`pyspark.sql.Window`), and Spark MLlib (`pyspark.ml`) -- not pandas
  relabeled as Spark, and not a Spark API surface with no Spark actually
  running underneath.
- **AWS Glue specifically has NOT been tested.** This project proves
  real, working Spark skills (DataFrame API, window functions, ML
  pipelines) on a local Spark cluster. AWS Glue is a managed Spark job
  runtime with its own job-authoring conventions (Glue jobs, Glue Data
  Catalog, `awsglue` library, DynamicFrames), and this project has not
  been run against it -- I don't have an AWS account with Glue access
  set up. I'm disclosing this honestly rather than implying Glue
  experience I don't have: what's genuinely demonstrated here is
  transferable Spark fundamentals, not hands-on Glue experience.

## Architecture

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

## Honest finding: a real bug, found and fixed

The first version of `price_model.build_feature_pipeline()` fed
`mileage_km` directly into `VectorAssembler` alongside the categorical
columns. It crashed with:

```
org.apache.spark.SparkException: Encountered null while assembling a row
with handleInvalid = "error". Consider removing nulls from dataset or
using handleInvalid = "keep" or "skip".
```

The cause: `VectorAssembler`'s default `handleInvalid="error"` does not
tolerate nulls in **numeric** input columns. I had assumed the
`StringIndexer(handleInvalid="keep")` stages already used for the
categorical columns covered this -- they don't; `StringIndexer`'s
`handleInvalid` option only applies to the categorical/string columns it
directly indexes, not to a separate numeric column feeding into the same
downstream assembler.

The deliberately-injected null `mileage_km` rows from
`generate_data.py` (~2% of rows) are exactly what surfaced this in
testing rather than in a real deployment.

**Fix**: added a real `pyspark.ml.feature.Imputer(strategy="median")`
stage before the assembler, producing `mileage_km_imputed`, which is what
actually feeds into `VectorAssembler`. I deliberately did *not* use
`VectorAssembler(handleInvalid="skip")` as the fix, even though it would
have "worked" -- that would silently drop real listings with missing
mileage from training and prediction, which is a worse outcome for a
pricing model than imputing a reasonable value. `tests/test_price_model.py::test_feature_pipeline_handles_null_mileage_without_crashing`
is a regression test guarding against this bug being silently
reintroduced.

## Real, verified results

Run live via `python run_pipeline.py` (full output also captured in the
test suite):

- Raw listings: 2,899 rows -> cleaned: 2,850 rows (49 duplicates removed,
  24 negative-price sign errors fixed, 60 missing-mileage rows flagged).
- Cross-market price gap vs. DE reference, X5 model: NL +7.6%, FR +3.73%,
  IT -4.9%, ES -7.34% -- these line up directionally and in rough
  magnitude with the injected `MARKET_PRICE_MULTIPLIER` config (NL=1.10,
  ES=0.92 vs DE=1.00), confirming the window-function/join-based
  cross-market analytics are computing the right thing, not just running
  without error.
- GBTRegressor price model vs. naive (market+model group mean) baseline:
  - Model RMSE ~1,235 EUR vs. naive RMSE ~5,920 EUR (**~79% improvement**)
  - Model MAE ~916 EUR vs. naive MAE ~4,875 EUR (**~81% improvement**)
  - Model R² ~0.984

A result this strong is expected given the synthetic price formula has
strong deterministic structure (a clean depreciation curve plus a small
noise term) -- a real used-car market would have more unexplained
variance. I'm reporting the number honestly rather than reframing it as
more impressive than it is: the point of this project is demonstrating a
correctly-built Spark ML pipeline and honest baseline comparison
methodology, not claiming this R² would hold on real market data.

## Running it

```
pip install -r requirements.txt
python run_pipeline.py     # end-to-end demo
pytest -q                  # full test suite (32 tests, real local Spark)
```

Requires a local Java installation (Java 8/11/17/21 all work with the
pinned PySpark version) -- Spark runs on the JVM even when driven from
Python.
