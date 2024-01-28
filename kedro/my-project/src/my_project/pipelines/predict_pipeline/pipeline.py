"""
This is a boilerplate pipeline 'predict_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

from my_project.pipelines.shared_nodes import tokenizer
from .nodes import predict_model


def create_predict_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(tokenizer, inputs=None, outputs="tokenizer", name="tokenizer"),
            node(
                predict_model,
                inputs={
                    "user_input": "user_input",  # Key in the Data Catalog for the user input
                    "tokenizer": "tokenizer",    # The output of the previous node (tokenizer)
                    "model": "model"             # Key in the Data Catalog for the model
                },
                outputs="predicted_label",
                name="predicted_label"
            ),
        ]
    )
