from kedro.pipeline import Pipeline, node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs=[
                "features_data",
                "params:ml.num_trees_grid",
                "params:ml.max_depth_grid",
                "params:ml.cv_folds",
                "params:split.train_ratio",
                "params:raw.models_dir",
            ],
            outputs="training_metrics",
            name="train_model_node",
        )
    ])
