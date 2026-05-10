import os
import logging
import mlflow
from pyspark.sql import DataFrame
from neuro_spark.core import (
    extract_feature_cols,
    split_by_subject,
    build_pipeline,
    train_with_cv,
    evaluate_model,
)

log = logging.getLogger(__name__)


def train_model(
    features_data: DataFrame,
    num_trees_grid: list,
    max_depth_grid: list,
    cv_folds: int,
    train_ratio: float,
    models_dir: str,
) -> dict:
    """Train a RandomForest with CrossValidator and track metrics in MLflow."""
    mlflow.set_experiment("neuro-spark-eeg-v2")
    mlflow.pyspark.ml.autolog()

    feature_cols = extract_feature_cols(features_data)
    train_df, test_df, train_subjects, test_subjects = split_by_subject(
        features_data, train_ratio
    )
    log.info("Train: %d epochs (%d subjects)", train_df.count(), len(train_subjects))
    log.info("Test: %d epochs (%d subjects)", test_df.count(), len(test_subjects))

    pipeline = build_pipeline(feature_cols)
    cv_model = train_with_cv(pipeline, train_df, num_trees_grid, max_depth_grid, cv_folds)

    result = evaluate_model(cv_model, test_df)
    metrics = result["metrics"]
    for name, value in metrics.items():
        log.info("  %s: %.4f", name, value)

    os.makedirs(models_dir, exist_ok=True)
    cv_model.write().overwrite().save(f"{models_dir}/cv_model")
    log.info("Model saved to %s/cv_model", models_dir)

    return metrics
