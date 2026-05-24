from kedro.pipeline import Pipeline, node
from .nodes import run_streaming_eeg


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=run_streaming_eeg,
            inputs=[
                "params:streaming.input_dir",
                "params:streaming.output_dir",
                "params:streaming.checkpoint",
                "params:streaming.parquet_dir",
                "params:streaming.model_cache",
                "params:streaming.timeout_seconds",
            ],
            outputs="streaming_stats",
            name="run_streaming_eeg_node",
        )
    ])
