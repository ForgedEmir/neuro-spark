"""
Pipeline d'entraînement : CrossValidator RandomForest avec MLflow.
Utilise les fonctions extraites du notebook poc_eeg.ipynb.
"""
import os
import sys
import json
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_spark.core import (
    create_spark_session,
    extract_feature_cols,
    split_by_subject,
    build_pipeline,
    train_with_cv,
    evaluate_model,
)

FEATURES_DIR = os.environ.get('FEATURES_DIR', '/opt/spark/data/features/')
MODELS_DIR = os.environ.get('MODELS_DIR', '/opt/spark/data/models/')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')
MLFLOW_TRACKING = os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/spark/mlruns')


def main():
    spark = create_spark_session()

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment('neuro-spark-eeg-v2')
    mlflow.pyspark.ml.autolog()

    # Chargement
    df = spark.read.parquet(FEATURES_DIR)
    print(f'Features chargées: {df.count():,} epochs')

    # Split + pipeline
    train_df, test_df, train_subjects, test_subjects = split_by_subject(df)
    feature_cols = extract_feature_cols(df)
    pipeline = build_pipeline(feature_cols)

    print(f'Train: {train_df.count():,} epochs ({len(train_subjects)} sujets)')
    print(f'Test:  {test_df.count():,} epochs ({len(test_subjects)} sujets)')

    # CrossValidator (4 combos × 3 folds = 12 entraînements)
    cv_model = train_with_cv(pipeline, train_df)

    # Évaluation
    result = evaluate_model(cv_model, test_df)
    predictions = result['predictions']
    metrics = result['metrics']

    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')

    # Sauvegarde
    os.makedirs(MODELS_DIR, exist_ok=True)
    cv_model.write().overwrite().save(f'{MODELS_DIR}/cv_model')

    os.makedirs('/opt/spark/data/metrics', exist_ok=True)
    with open('/opt/spark/data/metrics/training.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    spark.stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
