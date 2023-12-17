"""
This is a boilerplate pipeline 'predict_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

from my_project.pipelines.shared_nodes import tokenizer
from .nodes import predict_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(tokenizer, inputs=None, outputs="tokenizer", name="tokenizer"),
            node(predict_model, inputs={"user_input": "user_input", "tokenizer":"tokenizer", "model" : "model"}, outputs=None, name="prediction")
        ]
    )
