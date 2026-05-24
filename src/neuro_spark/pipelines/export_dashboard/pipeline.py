from kedro.pipeline import Pipeline, node
from .nodes import export_signal, export_features, export_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=export_signal,
            inputs=["parquet_data"],
            outputs="dashboard_signal",
            name="export_signal_node",
        ),
        node(
            func=export_features,
            inputs=["features_data"],
            outputs="dashboard_features",
            name="export_features_node",
        ),
        node(
            func=export_predictions,
            inputs=["predictions_data"],
            outputs="dashboard_predictions",
            name="export_predictions_node",
        ),
    ])
