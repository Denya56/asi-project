# src/my_project/pipelines/pycaret_pipeline

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_and_train_pycaret_automl, predict_with_best_model, compute_scoring

def create_pycaret_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                preprocess_and_train_pycaret_automl,
                ["raw_data", "parameters"],
                {"model_artifacts": "model_artifacts"},  # Update the output format
                name="preprocess_and_train_pycaret_automl_node",
            ),
            node(
                predict_with_best_model,
                ["raw_data", "model_artifacts"],  # Update to use "model_artifacts"
                "predictions",
                name="predict_with_best_model_node",
            ),
            node(
    		compute_scoring,
    		["predictions", "model_artifacts"],  # Update to use "model_artifacts"
    		"model_score",
	        name="compute_scoring_node",
	    ),
        ],
        tags=["pycaret"],
    )
