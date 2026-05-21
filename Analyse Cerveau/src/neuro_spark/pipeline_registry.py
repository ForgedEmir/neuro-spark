from kedro.pipeline import Pipeline

from neuro_spark.pipelines.ingestion.pipeline import create_pipeline as ingestion
from neuro_spark.pipelines.features.pipeline import create_pipeline as features
from neuro_spark.pipelines.training.pipeline import create_pipeline as training
from neuro_spark.pipelines.evaluation.pipeline import create_pipeline as evaluation
from neuro_spark.pipelines.export_dashboard.pipeline import create_pipeline as export_dashboard
from neuro_spark.pipelines.streaming_eeg.pipeline import create_pipeline as streaming_eeg


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "ingestion": ingestion(),
        "features": features(),
        "training": training(),
        "evaluation": evaluation(),
        "export": export_dashboard(),
        "streaming_eeg": streaming_eeg(),
    }
    # __default__ exclut le streaming (long-running, à lancer à la main)
    batch_pipelines = {
        k: v for k, v in pipelines.items() if k != "streaming_eeg"
    }
    pipelines["__default__"] = sum(batch_pipelines.values())
    return pipelines
