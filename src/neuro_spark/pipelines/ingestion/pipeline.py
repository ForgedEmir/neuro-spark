from kedro.pipeline import Pipeline, node
from .nodes import ingest_edf


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=ingest_edf,
            inputs=["params:raw.edf_dir", "params:raw.parquet_dir"],
            outputs="ingestion_stats",
            name="ingest_edf_node",
        )
    ])
