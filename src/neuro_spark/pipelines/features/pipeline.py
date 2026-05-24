from kedro.pipeline import Pipeline, node
from .nodes import build_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=build_features,
            inputs=["parquet_data", "params:features.t0_weight"],
            outputs="features_data",
            name="build_features_node",
        )
    ])
