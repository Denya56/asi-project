"""
This is a boilerplate pipeline 'main_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

from .nodes import hi, load_data, load_nlp


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_data, inputs=None, outputs="loaded_data", name="load_data")
        ]
    )
