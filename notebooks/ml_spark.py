"""Distributed ML: predict review star rating from text at full scale using
pyspark.ml (Tokenizer -> StopWordsRemover -> HashingTF -> IDF -> VectorAssembler
with numeric features -> multinomial LogisticRegression), trained across the
full ~7M-review dataset via Spark's local[*] executor, not a sample.

Complements python/ml/train_rating_model.py (scikit-learn on a 400K sample) --
this is the "does the distributed ML story hold up at scale" half.
"""
import json
import time
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer, VectorAssembler
from pyspark.sql import functions as F

from spark_session import get_spark
from io_utils import read_partitioned_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = PROJECT_ROOT / "reports" / "ml_spark_metrics.json"


def main():
    spark = get_spark("yelp-ml-spark")

    df = (
        read_partitioned_parquet(spark, PROCESSED_DIR / "review", "year", cast=int)
        .select("stars", "text", "text_len", "useful", "funny", "cool")
        .dropna(subset=["text"])
        .withColumn("stars", F.col("stars").cast("double"))
    )

    total = df.count()
    print(f"training on the full curated review dataset: {total:,} rows")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    print(f"train: {train_df.count():,}  test: {test_df.count():,}")

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=2 ** 16)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "text_len", "useful", "funny", "cool"],
        outputCol="features",
    )
    lr = LogisticRegression(
        featuresCol="features", labelCol="stars", family="multinomial", maxIter=30, regParam=0.01
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, assembler, lr])

    print("\nfitting Spark ML pipeline (Tokenizer -> StopWords -> HashingTF -> IDF -> LogisticRegression)...")
    start = time.time()
    model = pipeline.fit(train_df)
    fit_time = time.time() - start
    print(f"fit completed in {fit_time:.1f}s")

    predictions = model.transform(test_df)

    evaluator_acc = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="stars", predictionCol="prediction", metricName="f1")

    acc = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print(f"\nSpark ML accuracy: {acc:.4f}")
    print(f"Spark ML weighted F1: {f1:.4f}")

    predictions.groupBy("stars", "prediction").count().orderBy("stars", "prediction").show(30)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "model": "Spark ML: Tokenizer+StopWords+HashingTF(65536)+IDF+VectorAssembler+multinomial LogisticRegression",
                "train_rows": train_df.count(),
                "test_rows": test_df.count(),
                "fit_time_seconds": fit_time,
                "accuracy": acc,
                "weighted_f1": f1,
            },
            f,
            indent=2,
        )
    print(f"wrote metrics to {METRICS_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
