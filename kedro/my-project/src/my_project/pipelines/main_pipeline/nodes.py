import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from my_project.pipelines.shared_nodes import names, max_len, max_words


def hi():
    return "xdd"
    
def load_train_data():
    df = pd.read_csv('~/Documents/asi-project/data/train-data/train_dataset.csv', names=names)
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
    	             epochs=10,
    	             validation_data=(X_test, y_test))
    model3.save('~/Documents/asi-project/kedro/my-project/data/06_models/model')










