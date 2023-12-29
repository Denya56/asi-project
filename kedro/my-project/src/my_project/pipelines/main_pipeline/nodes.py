import pandas as pd
import tensorflow as tf
import keras
import optuna
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from my_project.pipelines.shared_nodes import names, max_len, max_words
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from optuna.integration import KerasPruningCallback
import pickle
# nodes.py
    
def hi():
    return "xdd"
    
def load_train_data():
    df = pd.read_csv('data/01_raw/train_dataset.csv-20231228T102241Z-001/train_dataset.csv', names=names)
    return df

def train_model(reviews, labels):
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.1, stratify=labels)

    model3 = Sequential()
    model3.add(layers.Embedding(max_words, 40, input_length=max_len))
    model3.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
    model3.add(layers.Dense(2,activation='softmax'))

    model3.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    #epochs changed to 1 from 10 (remember to change back)
    history = model3.fit(X_train,
    	             y_train,
    	             epochs=1,
    	             validation_data=(X_test, y_test))
    # Save the model using Kedro's catalog
    model_name = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"

    with open(f'data/06_models/{model_name}', 'wb') as file:
        pickle.dump(model3, file)
    #model3.save('~/ASI_2/asi-project/kedro/my-project/data/06_models/model.h5')
    
    return model3
def find_hyperparameters(reviews, labels, pretrained_model):
    # Define the objective function for Optuna
    def objective(trial):
        # Define the hyperparameters to optimize
        max_words = trial.suggest_int('max_words', 100, 1000)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.8)
        lstm_units = trial.suggest_int('lstm_units', 10, 50)

        # Load the pretrained model
        model = pretrained_model

        
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # epochs changed to 1 from 10 (remember to change back)
        history = model.fit(reviews, labels, epochs=1, validation_split=0.1, verbose=0, callbacks=[KerasPruningCallback(trial, 'accuracy')])

        # Evaluate the model on the validation set
        accuracy = history.history['accuracy'][-1]

        return accuracy

    # Create an Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial), n_trials=100)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    # You can use these hyperparameters as needed
    # ...

    return best_params
    
