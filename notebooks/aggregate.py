"""Spark aggregation jobs over the full curated dataset.

Produces the curated marts that feed the SQL layer, dashboard, and Tableau
extracts. Aggregation happens here (at big-data scale); downstream layers
only touch these already-small summary tables.
"""
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from spark_session import get_spark
from io_utils import write_via_pandas, read_partitioned_parquet, read_single_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MARTS_DIR = PROJECT_ROOT / "data" / "marts"


def main():
    spark = get_spark("yelp-aggregate")

    business = read_partitioned_parquet(spark, PROCESSED_DIR / "business", "state")
    review = read_partitioned_parquet(spark, PROCESSED_DIR / "review", "year", cast=int)
    checkin = read_single_parquet(spark, PROCESSED_DIR / "checkin")
    tip = read_single_parquet(spark, PROCESSED_DIR / "tip")

    business.createOrReplaceTempView("business")
    review.createOrReplaceTempView("review")
    checkin.createOrReplaceTempView("checkin")
    tip.createOrReplaceTempView("tip")

    # 1. Review volume + avg rating trend by year
    yearly_trend = spark.sql(
        """
        SELECT year, COUNT(*) AS review_count, ROUND(AVG(stars), 3) AS avg_stars
        FROM review
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
        """
    )

    # 2. Category-level stats (explode categories_arr)
    category_stats = spark.sql(
        """
        SELECT category,
               COUNT(DISTINCT business_id) AS business_count,
               ROUND(AVG(stars), 3) AS avg_stars,
               SUM(review_count) AS total_reviews
        FROM (
            SELECT business_id, stars, review_count, explode(categories_arr) AS category
            FROM business
            WHERE categories_arr IS NOT NULL
        )
        GROUP BY category
        HAVING business_count >= 50
        ORDER BY business_count DESC
        """
    )

    # 3. City-level stats
    city_stats = spark.sql(
        """
        SELECT city, state,
               COUNT(*) AS business_count,
               ROUND(AVG(stars), 3) AS avg_stars,
               SUM(review_count) AS total_reviews
        FROM business
        GROUP BY city, state
        HAVING business_count >= 20
        ORDER BY business_count DESC
        """
    )

    # 4. Business-level mart: business + checkin + tip counts, for SQL/dashboard/tableau
    business_mart = (
        business.join(checkin, "business_id", "left")
        .join(tip.groupBy("business_id").agg(F.count("*").alias("tip_count")), "business_id", "left")
        .withColumn("checkin_count", F.coalesce(F.col("checkin_count"), F.lit(0)))
        .withColumn("tip_count", F.coalesce(F.col("tip_count"), F.lit(0)))
    )

    # 5. Rating distribution
    rating_dist = review.groupBy("stars").count().orderBy("stars")

    outputs = {
        "yearly_trend": yearly_trend,
        "category_stats": category_stats,
        "city_stats": city_stats,
        "business_mart": business_mart,
        "rating_dist": rating_dist,
    }

    for name, df in outputs.items():
        print(f"\n=== {name} ===")
        df.show(10, truncate=60)
        out_path = MARTS_DIR / name
        row_count = write_via_pandas(df, out_path)
        pd.read_parquet(out_path / "part-0.parquet").to_csv(
            str(out_path) + ".csv", index=False
        )
        print(f"wrote {out_path} ({row_count:,} rows)")

    spark.stop()


if __name__ == "__main__":
    main()
