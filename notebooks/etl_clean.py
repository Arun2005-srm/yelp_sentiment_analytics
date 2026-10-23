"""Spark ETL: read raw Yelp JSON, clean/cast/dedupe, write curated Parquet.

Run against the full dataset (all ~7M reviews, ~150K businesses). This is
the "big data" layer: heavy lifting happens here, not in pandas.
"""
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from spark_session import get_spark
from io_utils import write_partitioned_via_pandas, write_via_pandas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_business(spark):
    df = spark.read.json(str(RAW_DIR / "yelp_academic_dataset_business.json"))
    df = (
        df.dropDuplicates(["business_id"])
        .withColumn("categories_arr", F.split(F.col("categories"), ",\\s*"))
        .withColumn("state", F.upper(F.trim(F.col("state"))))
        .withColumn("is_open", F.col("is_open").cast("boolean"))
        .select(
            "business_id", "name", "address", "city", "state", "postal_code",
            "latitude", "longitude", "stars", "review_count", "is_open",
            "categories", "categories_arr",
            # attributes/hours (nested Map<String,String>) deliberately dropped:
            # per-partition pandas/pyarrow round-tripping infers a different
            # STRUCT schema per state depending on which keys are present in
            # that state's data, which breaks unioning the partitions back
            # together. Unused downstream anyway.
        )
        .filter(F.col("business_id").isNotNull())
    )
    return df


def clean_review(spark):
    df = spark.read.json(str(RAW_DIR / "yelp_academic_dataset_review.json"))
    df = (
        df.dropDuplicates(["review_id"])
        .withColumn("stars", F.col("stars").cast(IntegerType()))
        .withColumn("date", F.to_timestamp("date"))
        .withColumn("year", F.year("date"))
        .withColumn("text_len", F.length("text"))
        .filter(F.col("stars").between(1, 5))
        .select(
            "review_id", "user_id", "business_id", "stars", "useful", "funny",
            "cool", "text", "text_len", "date", "year",
        )
    )
    return df


def clean_user(spark):
    df = spark.read.json(str(RAW_DIR / "yelp_academic_dataset_user.json"))
    df = (
        df.dropDuplicates(["user_id"])
        .withColumn("yelping_since", F.to_timestamp("yelping_since"))
        .withColumn("friend_count", F.size(F.split(F.col("friends"), ",\\s*")))
        .select(
            "user_id", "name", "review_count", "yelping_since", "useful",
            "funny", "cool", "fans", "average_stars", "friend_count", "elite",
        )
    )
    return df


def clean_checkin(spark):
    df = spark.read.json(str(RAW_DIR / "yelp_academic_dataset_checkin.json"))
    df = df.withColumn("checkin_count", F.size(F.split(F.col("date"), ",\\s*"))).select(
        "business_id", "checkin_count"
    )
    return df


def clean_tip(spark):
    df = spark.read.json(str(RAW_DIR / "yelp_academic_dataset_tip.json"))
    df = df.withColumn("date", F.to_timestamp("date")).select(
        "user_id", "business_id", "text", "date", "compliment_count"
    )
    return df


def main():
    spark = get_spark("yelp-etl-clean")

    jobs = {
        "business": clean_business,
        "review": clean_review,
        "user": clean_user,
        "checkin": clean_checkin,
        "tip": clean_tip,
    }

    for name, fn in jobs.items():
        print(f"\n=== cleaning {name} ===")
        df = fn(spark)
        count = df.count()
        print(f"{name}: {count:,} rows")
        df.show(3, truncate=80)

        out_path = PROCESSED_DIR / name
        if name == "review":
            write_partitioned_via_pandas(df, out_path, "year")
        elif name == "business":
            write_partitioned_via_pandas(df, out_path, "state")
        else:
            write_via_pandas(df, out_path)
        print(f"wrote {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()
