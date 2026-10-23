"""Shared SparkSession factory with embedded Hive support.

Uses Spark's built-in Hive integration (local Derby metastore) so HiveQL
(CREATE EXTERNAL TABLE, PARTITIONED BY, MSCK REPAIR TABLE, etc.) runs for
real without standing up a separate Hadoop/Hive cluster. In production this
would point at a cluster-hosted Hive metastore (EMR/Dataproc/on-prem).
"""
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"


def get_spark(app_name: str = "yelp-analytics", shuffle_partitions: int = 8) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.driver.memory", "10g")
        .config("spark.ui.enabled", "false")
        .enableHiveSupport()
        .getOrCreate()
    )
