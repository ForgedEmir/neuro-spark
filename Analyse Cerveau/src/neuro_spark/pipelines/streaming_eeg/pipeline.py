from kedro.pipeline import Pipeline, node
from .nodes import run_streaming_eeg


def create_pipeline(**_) -> Pipeline:
    return Pipeline([
        node(
            func=run_streaming_eeg,
            inputs=dict(
                stream_input="params:streaming_eeg.input_dir",
                stream_output="params:streaming_eeg.output_dir",
                checkpoint="params:streaming_eeg.checkpoint_dir",
                parquet_dir="params:raw.parquet_dir",
                model_cache="params:streaming_eeg.model_cache",
                timeout_seconds="params:streaming_eeg.timeout_seconds",
            ),
            outputs="streaming_eeg_metrics",
            name="run_streaming_eeg",
        ),
    ])
