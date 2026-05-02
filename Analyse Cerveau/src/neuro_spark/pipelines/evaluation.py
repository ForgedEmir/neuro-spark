"""
Pipeline d'évaluation : métriques détaillées sur le test set.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_spark.core import create_spark_session, extract_feature_cols, split_by_subject, evaluate_model
from pyspark.ml.tuning import CrossValidatorModel

FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
MODELS_DIR = os.environ.get('MODELS_DIR', '/opt/spark/data/models/')
PREDICTIONS_DIR = os.environ.get('PREDICTIONS_DIR', '/opt/spark/data/predictions/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')


def main():
    spark = create_spark_session()

    df = spark.read.parquet(FEATURES_DIR)
    _, test_df, _, _ = split_by_subject(df)

    cv_model = CrossValidatorModel.load(f'{MODELS_DIR}/cv_model')
    result = evaluate_model(cv_model, test_df)

    # Sauvegarde prédictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    result['predictions'].select('subject_id', 'task_label', 'prediction') \
        .coalesce(1) \
        .write.mode('overwrite').parquet(PREDICTIONS_DIR)

    # Métriques
    os.makedirs('/opt/spark/data/metrics', exist_ok=True)
    with open('/opt/spark/data/metrics/evaluation.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    print('Évaluation:')
    for name, value in result['metrics'].items():
        print(f'  {name}: {value:.4f}')

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
