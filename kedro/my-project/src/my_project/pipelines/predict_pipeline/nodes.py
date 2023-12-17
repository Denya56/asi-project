"""
This is a boilerplate pipeline 'predict_pipeline'
generated using Kedro 0.18.14
"""
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from my_project.pipelines.shared_nodes import names, max_len, max_words
from keras.utils import pad_sequences

def predict_model(user_input, model, tokenizer):
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_data = pad_sequences(user_sequence, maxlen=max_len)
    # Make predictions for the user input
    user_prediction = model.predict(user_data)[0]
    # Print the predicted label for the user input
    if user_prediction[1] > user_prediction[0]:
        predicted_label = 1
    else:
        predicted_label = 0

    return predicted_label
