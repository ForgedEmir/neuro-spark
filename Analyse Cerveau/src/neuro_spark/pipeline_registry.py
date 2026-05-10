from kedro.pipeline import Pipeline

from neuro_spark.pipelines.ingestion.pipeline import create_pipeline as ingestion
from neuro_spark.pipelines.features.pipeline import create_pipeline as features
from neuro_spark.pipelines.training.pipeline import create_pipeline as training
from neuro_spark.pipelines.evaluation.pipeline import create_pipeline as evaluation
from neuro_spark.pipelines.export_dashboard.pipeline import create_pipeline as export_dashboard


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "ingestion": ingestion(),
        "features": features(),
        "training": training(),
        "evaluation": evaluation(),
        "export": export_dashboard(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
