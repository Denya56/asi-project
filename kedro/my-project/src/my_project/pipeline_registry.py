"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from my_project.pipelines.main_pipeline.pipeline import create_pipeline
from my_project.pipelines.pycaret_pipeline import create_pycaret_pipeline
from my_project.pipelines.predict_pipeline import create_predict_pipeline
def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Create the main and PyCaret pipelines
    main_pipeline = create_pipeline()
    pycaret_pipeline = create_pycaret_pipeline()
    predict_pipeline = create_predict_pipeline()

    # Combine the pipelines
    pipelines = {
        "main": main_pipeline,
        "pycaret": pycaret_pipeline,
        "predict_pipeline": predict_pipeline
    }

    # Add any other pipelines you might have in the future
    # Example:
    # additional_pipeline = create_additional_pipeline()
    # pipelines["additional"] = additional_pipeline

    return pipelines
