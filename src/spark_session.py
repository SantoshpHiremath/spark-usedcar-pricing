"""
Shared SparkSession factory. Runs Spark in local mode (`local[*]`) --
this environment has no real Spark cluster or AWS Glue job runtime, so
every test and demo run here executes on a genuine, real, local Spark
engine (real JVM, real DataFrame execution, real Catalyst query
planning), just not distributed across a cluster. See README for the
full disclosure on how this differs from AWS Glue specifically.
"""
from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "used_car_pricing_analytics") -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")  # small, since this is local-mode, not a real cluster
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
