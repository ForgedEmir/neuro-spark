from kedro.pipeline import Pipeline, node
from .nodes import evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=evaluate,
            inputs=[
                "features_data",
                "params:raw.models_dir",
                "params:split.train_ratio",
            ],
            outputs=["predictions_data", "evaluation_metrics"],
            name="evaluate_node",
        )
    ])
