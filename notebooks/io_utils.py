"""Parquet materialization helpers that bypass Spark's native DataFrameWriter.

Spark's built-in writer routes through Hadoop's FileOutputCommitter, which on
Windows calls into a native NativeIO.Windows.access0() JNI method backed by
hadoop.dll. That DLL isn't bundled with pip-installed PySpark and there's no
legitimate first-party source for it on Windows -- only unofficial third-party
binary mirrors, which we deliberately avoid downloading and executing.

Spark still does all the actual heavy lifting (reading multi-GB JSON,
filtering, casting, deduplication, joins, aggregation, Spark SQL). Only the
final "write DataFrame to Parquet" step is done by collecting each partition
slice to the driver and writing it with pyarrow directly, which never touches
Hadoop's committer. Partition directories are named Hive-style
(`col=value/`) so the output is layout-compatible with Spark/Hive external
tables reading it back.
"""
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F


def write_partitioned_via_pandas(df, out_dir: Path, partition_col: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.cache()
    values = [row[0] for row in df.select(partition_col).distinct().collect() if row[0] is not None]
    print(f"  writing {len(values)} partitions to {out_dir} (partition col: {partition_col})")

    total_rows = 0
    for v in sorted(values, key=str):
        part_dir = out_dir / f"{partition_col}={v}"
        part_dir.mkdir(parents=True, exist_ok=True)
        pdf = df.filter(df[partition_col] == v).drop(partition_col).toPandas()
        pdf.to_parquet(part_dir / "part-0.parquet", engine="pyarrow", index=False)
        total_rows += len(pdf)
        print(f"    {partition_col}={v}: {len(pdf):,} rows")
    df.unpersist()
    print(f"  total: {total_rows:,} rows")
    return total_rows


def read_partitioned_parquet(spark, base_dir: Path, partition_col: str, cast=None):
    """Read back a Hive-style-partitioned Parquet dataset written by
    write_partitioned_via_pandas, without asking Spark/Hadoop to list the
    parent directory (see module docstring -- that listing also needs the
    missing native Windows Hadoop library). Partition subdirectories are
    discovered with plain Python pathlib, each file is read individually
    (proven to work), and the partition column is reattached from the
    directory name before unioning everything together.
    """
    base_dir = Path(base_dir)
    part_dirs = sorted(base_dir.glob(f"{partition_col}=*"))
    if not part_dirs:
        raise FileNotFoundError(f"no {partition_col}=* partitions found under {base_dir}")

    frames = []
    for part_dir in part_dirs:
        raw_value = part_dir.name.split("=", 1)[1]
        value = cast(raw_value) if cast else raw_value
        file_path = part_dir / "part-0.parquet"
        part_df = spark.read.parquet(str(file_path)).withColumn(partition_col, F.lit(value))
        frames.append(part_df)

    result = frames[0]
    for f in frames[1:]:
        result = result.unionByName(f)
    return result


def read_single_parquet(spark, out_dir: Path):
    """Read back a single-file (unpartitioned) table written by
    write_via_pandas, reading the concrete file directly rather than
    listing the parent directory (same native-library reason as above)."""
    return spark.read.parquet(str(Path(out_dir) / "part-0.parquet"))


def write_via_pandas(df, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.toPandas()
    pdf.to_parquet(out_dir / "part-0.parquet", engine="pyarrow", index=False)
    print(f"  wrote {len(pdf):,} rows to {out_dir}")
    return len(pdf)
