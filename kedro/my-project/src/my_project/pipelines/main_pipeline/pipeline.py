"""
This is a boilerplate pipeline 'main_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

from my_project.pipelines.shared_nodes import load_nlp, tokenizer, prepare_data
from .nodes import hi, load_train_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_nlp, inputs=None, outputs="nlp", name="nlp"),
            node(tokenizer, inputs=None, outputs="tokenizer", name="tokenizer"),
            node(load_train_data, inputs=None, outputs="df", name="df"),
            node(prepare_data, inputs={"df": "df", "nlp": "nlp", "tokenizer": "tokenizer"}, outputs=["labels", "reviews"], name="data_prepared")
        ]
    )
