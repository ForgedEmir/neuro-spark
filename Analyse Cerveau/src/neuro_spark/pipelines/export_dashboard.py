"""
Export des données pour le dashboard Dash.
Utilise spark.write.parquet() — pas de toPandas().
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_spark.core import create_spark_session, clean_channel_names

PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')
FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', '/opt/spark/data/predictions/')
DASHBOARD_DIR = os.environ.get('DASHBOARD_DIR', '/opt/spark/data/dashboard/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')


def main():
    spark = create_spark_session()

    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    meta_fields = [
        StructField("subject_id", StringType(), True),
        StructField("run_id", StringType(), True),
        StructField("time", DoubleType(), True),
        StructField("task_label", StringType(), True),
    ]
    schema = StructType(meta_fields + [StructField("Cz..", DoubleType(), True)])

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    # Signal brut Cz (S001/R03, 20 secondes)
    raw_df = spark.read.schema(schema).parquet(PARQUET_DIR)
    raw_df = clean_channel_names(raw_df)

    raw_df.filter(
        (F.col('subject_id') == 'S001') & (F.col('run_id') == 'R03')
    ).select('time', 'Cz', 'task_label') \
      .orderBy('time').limit(3200) \
      .coalesce(1) \
      .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/signal_sample.parquet')

    # Features
    spark.read.parquet(FEATURES_DIR) \
        .coalesce(1) \
        .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/features.parquet')

    # Prédictions
    spark.read.parquet(PREDICTIONS_DIR) \
        .coalesce(1) \
        .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/predictions.parquet')

    print(f'✓ Dashboard data exporté dans {DASHBOARD_DIR}')

    spark.stop()
    return 0


if __name__ == '__main__':
    import pyspark.sql.functions as F
    sys.exit(main())
