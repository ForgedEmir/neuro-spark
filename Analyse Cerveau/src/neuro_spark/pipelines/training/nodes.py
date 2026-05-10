import os
import mlflow
from pyspark.sql import DataFrame
from neuro_spark.core import (
    extract_feature_cols,
    split_by_subject,
    build_pipeline,
    train_with_cv,
    evaluate_model,
)


def train_model(
    features_data: DataFrame,
    num_trees_grid: list,
    max_depth_grid: list,
    cv_folds: int,
    train_ratio: float,
    models_dir: str,
) -> dict:
    """CrossValidator RandomForest + tracking MLflow."""
    mlflow.set_experiment("neuro-spark-eeg-v2")
    mlflow.pyspark.ml.autolog()

    feature_cols = extract_feature_cols(features_data)
    train_df, test_df, train_subjects, test_subjects = split_by_subject(
        features_data, train_ratio
    )
    print(f"Train : {train_df.count():,} epochs ({len(train_subjects)} sujets)")
    print(f"Test  : {test_df.count():,} epochs ({len(test_subjects)} sujets)")

    pipeline = build_pipeline(feature_cols)
    cv_model = train_with_cv(pipeline, train_df, num_trees_grid, max_depth_grid, cv_folds)

    result = evaluate_model(cv_model, test_df)
    metrics = result["metrics"]
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    cv_model.write().overwrite().save(f"{models_dir}/cv_model")
    print(f"✓ Modèle sauvegardé dans {models_dir}/cv_model")

    return metrics
