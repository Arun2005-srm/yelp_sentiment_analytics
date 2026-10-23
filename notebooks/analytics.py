"""Deep Spark SQL analytics over the full curated dataset: window functions,
multi-table joins, and a real partition-pruning performance benchmark.

The benchmark compares reading only the relevant year=2021 partition file
directly against reading and unioning every file in an equivalent flat,
non-partitioned copy before filtering. This is the "why partitioning matters
at scale" story that was originally going to be told via Hive. Note both
sides read explicit, known file paths rather than a parent directory --
Spark/Hadoop's own directory-listing-based partition discovery needs the
same native Windows Hadoop library that's missing here (see io_utils.py),
so this benchmark demonstrates the partitioning *principle* (touch only the
data you need vs. touch all of it) via manual file selection rather than
Spark's automatic partition pruning end-to-end.
"""
import shutil
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_session import get_spark
from io_utils import read_partitioned_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FLAT_DIR = PROJECT_ROOT / "data" / "processed_flat"
RESULTS_DIR = PROJECT_ROOT / "reports" / "spark_analytics_results"


def build_flat_copy():
    """Plain file copies (no Spark job) of the year-partitioned review
    Parquet into a single flat directory, for the pruning benchmark."""
    flat_review = FLAT_DIR / "review"
    flat_review.mkdir(parents=True, exist_ok=True)
    partitions = sorted((PROCESSED_DIR / "review").glob("year=*"))
    for part in partitions:
        year = part.name.split("=")[1]
        src = part / "part-0.parquet"
        dst = flat_review / f"part_{year}.parquet"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    print(f"flat copy ready: {len(list(flat_review.glob('*.parquet')))} files in {flat_review}")


def window_function_analytics(spark):
    business = read_partitioned_parquet(spark, PROCESSED_DIR / "business", "state")
    review = read_partitioned_parquet(spark, PROCESSED_DIR / "review", "year", cast=int)
    business.createOrReplaceTempView("business")
    review.createOrReplaceTempView("review")

    print("\n=== Top 5 businesses by review_count per state (RANK window fn) ===")
    top_per_state = spark.sql(
        """
        SELECT state, name, review_count, stars, rnk FROM (
            SELECT state, name, review_count, stars,
                   RANK() OVER (PARTITION BY state ORDER BY review_count DESC) AS rnk
            FROM business
        ) ranked
        WHERE rnk <= 5
        ORDER BY state, rnk
        """
    )
    top_per_state.show(20, truncate=False)
    top_per_state.toPandas().to_csv(RESULTS_DIR / "top_businesses_per_state.csv", index=False)

    print("\n=== Year-over-year review growth (LAG window fn) ===")
    yoy = spark.sql(
        """
        SELECT year, review_count, avg_stars,
               review_count - LAG(review_count) OVER (ORDER BY year) AS yoy_growth,
               ROUND(100.0 * (review_count - LAG(review_count) OVER (ORDER BY year))
                     / LAG(review_count) OVER (ORDER BY year), 2) AS yoy_pct_change
        FROM (
            SELECT year, COUNT(*) AS review_count, ROUND(AVG(stars), 3) AS avg_stars
            FROM review WHERE year IS NOT NULL GROUP BY year
        )
        ORDER BY year
        """
    )
    yoy.show(20)
    yoy.toPandas().to_csv(RESULTS_DIR / "yoy_review_growth.csv", index=False)

    print("\n=== States with widest rating spread (>=100-review businesses) ===")
    spread = spark.sql(
        """
        SELECT state, MAX(stars) AS max_stars, MIN(stars) AS min_stars,
               MAX(stars) - MIN(stars) AS spread, COUNT(*) AS qualifying_businesses
        FROM business
        WHERE review_count >= 100
        GROUP BY state
        HAVING qualifying_businesses >= 10
        ORDER BY spread DESC
        """
    )
    spread.show(20)
    spread.toPandas().to_csv(RESULTS_DIR / "rating_spread_by_state.csv", index=False)

    print("\n=== Running total of reviews per user, ranked by activity (window fn) ===")
    review_user = review.select("user_id", "stars")
    user_activity = (
        review_user.groupBy("user_id")
        .agg(F.count("*").alias("review_count"), F.avg("stars").alias("avg_stars_given"))
        .withColumn("rank", F.rank().over(Window.orderBy(F.desc("review_count"))))
        .filter(F.col("rank") <= 20)
        .orderBy("rank")
    )
    user_activity.show(20)
    user_activity.toPandas().to_csv(RESULTS_DIR / "most_active_users.csv", index=False)


def partition_pruning_benchmark(spark):
    """Compares reading only the year=2021 partition file directly (what
    Spark's own partition pruning would do, if directory-based partition
    discovery worked on this machine -- see module docstring) against
    reading and unioning every flat file before filtering (a true full
    scan). Both sides read known, explicit file paths -- no directory
    listing on either side, since that's the operation that's broken here,
    not the thing being benchmarked.
    """
    print("\n=== Partition pruning benchmark: year=2021 filter ===")

    pruned_file = PROCESSED_DIR / "review" / "year=2021" / "part-0.parquet"
    start = time.time()
    partitioned_count = spark.read.parquet(str(pruned_file)).count()
    partitioned_time = time.time() - start
    print(f"Pruned read (single year=2021 file only): {partitioned_count:,} rows in {partitioned_time:.2f}s")

    flat_files = sorted((FLAT_DIR / "review").glob("*.parquet"))
    start = time.time()
    flat_df = spark.read.parquet(str(flat_files[0]))
    for fp in flat_files[1:]:
        flat_df = flat_df.unionByName(spark.read.parquet(str(fp)))
    flat_count = flat_df.filter(F.year("date") == 2021).count()
    flat_time = time.time() - start
    print(f"Full scan (all {len(flat_files)} flat files, then filter): {flat_count:,} rows in {flat_time:.2f}s")

    assert partitioned_count == flat_count, "row counts should match between pruned and full-scan reads"

    speedup = flat_time / partitioned_time if partitioned_time > 0 else float("inf")
    result = (
        f"Pruned (single partition file, year=2021):        {partitioned_time:.2f}s, {partitioned_count:,} rows\n"
        f"Full scan ({len(flat_files)} flat files, then filter): {flat_time:.2f}s, {flat_count:,} rows\n"
        f"Speedup from partitioning: {speedup:.2f}x\n"
    )
    print(result)
    (RESULTS_DIR / "partition_pruning_benchmark.txt").write_text(result)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    build_flat_copy()

    spark = get_spark("yelp-analytics-deep")
    window_function_analytics(spark)
    partition_pruning_benchmark(spark)
    spark.stop()


if __name__ == "__main__":
    main()
