# src/my_project/pipelines/pycaret_pipeline

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, train_pycaret_automl

def create_pycaret_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_data,
                ["raw_data", "parameters"],
                ["preprocessed_data"],
                name="preprocess_data_node",
            ),
            node(
                train_pycaret_automl,
                ["preprocessed_data", "parameters"],
                ["model", "model_artifacts"],
                name="train_pycaret_automl_node",
            ),
        ],
        tags=["pycaret"],
    )
