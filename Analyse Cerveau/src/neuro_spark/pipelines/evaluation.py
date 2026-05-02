"""
Pipeline d'évaluation : métriques détaillées et matrice de confusion.
"""
import os
import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
MODELS_DIR = os.environ.get('MODELS_DIR', '/opt/spark/data/models/')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', '/opt/spark/data/predictions/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')
TEST_SUBJECTS = [f'S{i:03d}' for i in range(53, 67)]


def main():
    spark = SparkSession.builder \
        .appName('EEG-Evaluation') \
        .master(SPARK_MASTER) \
        .getOrCreate()

    # Charger le modèle et les features de test
    from pyspark.ml import PipelineModel
    model = PipelineModel.load(f'{MODELS_DIR}/rf_model')
    df = spark.read.parquet(FEATURES_DIR)
    test_df = df.filter(F.col('subject_id').isin(TEST_SUBJECTS))

    # Prédictions
    predictions = model.transform(test_df)

    # Métriques globales
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

    results = {}
    for metric in ['accuracy', 'f1', 'weightedPrecision', 'weightedRecall']:
        evaluator.setMetricName(metric)
        results[metric] = evaluator.evaluate(predictions)

    # Sauvegarde
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    predictions.select('task_label', 'prediction').coalesce(1) \
        .write.mode('overwrite').parquet(PREDICTIONS_DIR)

    with open('/opt/spark/data/metrics/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Évaluation — Accuracy: {results["accuracy"]:.4f}, F1: {results["f1"]:.4f}')
    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
