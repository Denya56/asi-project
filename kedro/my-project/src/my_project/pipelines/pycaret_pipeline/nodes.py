import pandas as pd
import tensorflow as tf
import keras
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from keras.models import Sequential
from keras import layers
from my_project.pipelines.shared_nodes import names, max_len, max_words
from datetime import datetime
from pycaret.classification import *
from kedro.extras.datasets.pandas import CSVDataSet  # Import CSVDataSet
from pycaret.classification import predict_model, evaluate_model
from pycaret.classification import setup
import pickle
# /pycaret_pipeline/nodes.py

      
def preprocess_and_train_pycaret_automl(raw_data, parameters):
    # Assuming raw_data is a DataFrame with unnamed target column
    
    # Rename the target column to 'label'
    raw_data = raw_data.rename(columns={raw_data.columns[-1]: 'label'})

    # Set up PyCaret environment
    setup(
        data=raw_data,
        target='label',  # Specify the target column explicitly
        preprocess=True,
        session_id=123,
        log_experiment=True,
        experiment_name='text_classification_experiment'
    )

    # Compare models and select the best one
    best_model = compare_models()

    # Create and tune the best model
    tuned_model = create_model(best_model)
    
    # Specify the directory where you want to save the model
    model_save_directory = "data/09_pycaretmodels"

    # Create the directory if it doesn't exist
    os.makedirs(model_save_directory, exist_ok=True)

    # Temporarily set the working directory to the specified directory
    current_working_directory = os.getcwd()
    os.chdir(model_save_directory)

    # Save the tuned model
    saved_model_path = save_model(tuned_model, model_name='tuned_model')

    # Reset the working directory to its original value
    os.chdir(current_working_directory)

    # Return the trained model and its artifacts, including the saved model path
    model_artifacts = {
        "best_model": best_model,
        "tuned_model": tuned_model,
        "saved_model_path": saved_model_path,
    }
    
    # Print the saved_model_path for debugging
    print(f"Saved model path: {saved_model_path}")
     # Print the contents of the model_artifacts dictionary for debugging
    print("Model Artifacts:")
    print(model_artifacts)

    # Return the trained model and its artifacts
    return {"model_artifacts": model_artifacts}
    

def compute_scoring(predictions, model_artifacts):
    try:
        # Assuming 'predictions' is a dictionary with a key 'predictions'
        predictions_df = predictions['predictions']

      	
        y_true = predictions_df.iloc[:, 1]  
        y_pred = predictions_df.iloc[:, 2]  

        # Extract the best model from model_artifacts
        best_model = model_artifacts["tuned_model"]
        print("Best model:")
        print(best_model)

        # Calculate precision score
        score = precision_score(y_true, y_pred)
        print(f"Precision Score: {score}")

        return {"model_score": score}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
def predict_with_best_model(raw_data, parameters):
    # Specify the directory where the model is saved
    model_save_directory = "data/09_pycaretmodels"

    # Load the saved model
    loaded_model = load_model(os.path.join(model_save_directory, 'tuned_model'))

    # Use the best model to make predictions on the raw_data
    predictions = predict_model(loaded_model, data=raw_data)

    return {"predictions": predictions}
    
def train_pycaret_automl(preprocessed_data, parameters):
    return "????"
