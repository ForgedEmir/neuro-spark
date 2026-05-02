"""
Export des données pour le dashboard Dash.
Utilise spark.write.parquet() — pas de toPandas().
"""
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', '/opt/spark/data/predictions/')
DASHBOARD_DIR = os.environ.get('DASHBOARD_DIR', '/opt/spark/data/dashboard/')
PARQUET_DIR = os.environ.get('PARQUET_DIR', '/opt/spark/data/parquet/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')


def main():
    spark = SparkSession.builder \
        .appName('EEG-Dashboard-Export') \
        .master(SPARK_MASTER) \
        .getOrCreate()

    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    # ── Signal brut Cz (S001/R03, 20 secondes) ──
    # Lecture avec schéma explicite
    meta_fields = [
        StructField('subject_id', StringType(), True),
        StructField('run_id', StringType(), True),
        StructField('time', DoubleType(), True),
        StructField('task_label', StringType(), True),
    ]
    canal_names = ['Cz..']  # Seulement Cz pour l'export
    canal_fields = [StructField(name, DoubleType(), True) for name in canal_names]
    schema = StructType(meta_fields + canal_fields)

    raw_df = spark.read.schema(schema).parquet(PARQUET_DIR)
    # Renommage
    for c in raw_df.columns:
        if '.' in c:
            raw_df = raw_df.withColumnRenamed(c, c.replace('.', ''))

    raw_df.filter(
        (F.col('subject_id') == 'S001') & (F.col('run_id') == 'R03')
    ).select('time', 'Cz', 'task_label') \
      .orderBy('time').limit(3200) \
      .coalesce(1) \
      .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/signal_sample.parquet')

    # ── Features ──
    df_features = spark.read.parquet(FEATURES_DIR)
    df_features.coalesce(1) \
        .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/features.parquet')

    # ── Prédictions ──
    predictions = spark.read.parquet(PREDICTIONS_DIR)
    predictions.select('task_label', 'prediction') \
        .coalesce(1) \
        .write.mode('overwrite').parquet(f'{DASHBOARD_DIR}/predictions.parquet')

    print(f'Données exportées dans {DASHBOARD_DIR}')
    print('  signal_sample.parquet | features.parquet | predictions.parquet')

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
