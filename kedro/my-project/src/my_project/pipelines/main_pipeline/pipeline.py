"""
This is a boilerplate pipeline 'main_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

from .nodes import hi, load_data, load_nlp, apply_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_data, inputs=None, outputs="data", name="data"),
            node(load_nlp, inputs=None, outputs="nlp", name="nlp"),
            node(apply_data, inputs={"data": "data", "nlp": "nlp"}, outputs="data_applied")
        ]
    )
