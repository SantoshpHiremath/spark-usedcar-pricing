"""
Shared SparkSession factory. Runs Spark in local mode (`local[*]`) --
every test and demo run executes on a genuine local Spark engine (real
JVM, real DataFrame execution, real Catalyst query planning), not
distributed across a cluster. See README for notes on running this
code on a managed runtime such as AWS Glue.
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
