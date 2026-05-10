from pyspark.sql import DataFrame
from pyspark.ml.tuning import CrossValidatorModel
from neuro_spark.core import split_by_subject, evaluate_model


def evaluate(
    features_data: DataFrame, models_dir: str, train_ratio: float
) -> tuple[DataFrame, dict]:
    """Charge le modèle et évalue sur le test set."""
    _, test_df, _, _ = split_by_subject(features_data, train_ratio)

    cv_model = CrossValidatorModel.load(f"{models_dir}/cv_model")
    result = evaluate_model(cv_model, test_df)

    print("Évaluation :")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")

    predictions = result["predictions"].select("subject_id", "task_label", "prediction")
    return predictions, result["metrics"]
