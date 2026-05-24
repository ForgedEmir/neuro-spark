import logging
from pyspark.sql import DataFrame
from pyspark.ml.tuning import CrossValidatorModel
from neuro_spark.core import split_by_subject, evaluate_model

log = logging.getLogger(__name__)


def evaluate(
    features_data: DataFrame, models_dir: str, train_ratio: float
) -> tuple[DataFrame, dict]:
    """Load the saved model and evaluate it on the held-out test set."""
    _, test_df, _, _ = split_by_subject(features_data, train_ratio)

    cv_model = CrossValidatorModel.load(f"{models_dir}/cv_model")
    result = evaluate_model(cv_model, test_df)

    for name, value in result["metrics"].items():
        log.info("  %s: %.4f", name, value)

    predictions = result["predictions"].select("subject_id", "task_label", "prediction")
    return predictions, result["metrics"]
