import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from neuro_spark.core import clean_channel_names


def export_signal(parquet_data: DataFrame) -> DataFrame:
    """Signal brut Cz pour le sujet S001 run R03 (20 secondes)."""
    df = clean_channel_names(parquet_data)
    return (
        df.filter((F.col("subject_id") == "S001") & (F.col("run_id") == "R03"))
        .select("time", "Cz", "task_label")
        .orderBy("time")
        .limit(3200)
    )


def export_features(features_data: DataFrame) -> DataFrame:
    return features_data


def export_predictions(predictions_data: DataFrame) -> DataFrame:
    return predictions_data
