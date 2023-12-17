import pathlib
import os
import pickle
from typing import Any, Iterable, Annotated

from fastapi import Depends, FastAPI, Query, Body
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

app = FastAPI(
    title="FastAPI + Kedro",
    version="0.0.1",
    license_info={
        "name": "GNU GENERAL PUBLIC LICENSE",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    },
)

def get_session() -> Iterable[KedroSession]:
    bootstrap_project(pathlib.Path().cwd())
    with KedroSession.create() as session:
        yield session

def get_context(session: KedroSession = Depends(get_session)) -> Iterable[KedroContext]:
    yield session.load_context()

@app.get("/models")
def get_trained_model_names(
    session: KedroSession = Depends(get_session),
    context: KedroContext = Depends(get_context),
) -> dict[str, Any]:
    models_path = context.project_path / "data/06_models"
    trained_model_names = [
        model.stem for model in models_path.glob("*.pkl") if model.is_file()
    ]
    return {" trained_model_names": str(trained_model_names)}

def load_model(context: KedroContext, model_name: str) -> Any:
    model_path = context.project_path / "data/06_models" / (model_name + ".pkl")
    if not model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model
    
@app.post("/predict")
def predict(
    model_name: Annotated[str, Query()],
    input_data: Annotated[list[str], Body()],
    session: KedroSession = Depends(get_session),
    context: KedroContext = Depends(get_context),
) -> Any:
    loaded_model = load_model(context, model_name)
    pipeline_result = session.run(
        pipeline_name="predict_pipeline",
    )
    
    return {"model_name": model_name, "input_data": input_data, "prediction": pipeline_result}
