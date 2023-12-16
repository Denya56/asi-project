"""
This is a boilerplate pipeline 'main_pipeline'
generated using Kedro 0.18.14
"""
import pandas as pd
import spacy
import re
import tensorflow as tf
import keras
import jrdpackage as jrd
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import pad_sequences, to_categorical


def hi():
    return "xdd"
    
def load_data():
    names=['text', 'label']
    df = pd.read_csv('~/Documents/asi-project/data/train-data/train_dataset.csv', names=names)
    data=df[['text','label']]
    re_letters=re.compile(r"[^a-zA-Z\s']")
    jrd.clean_data(data, re_letters)
    return data
    
def load_nlp():
    try:
        nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
    except OSError:
        # Download the model if it's not already available
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
    nlp.add_pipe('sentencizer')
    nlp.Defaults.stop_words.add("game")
    nlp.Defaults.stop_words.add("play")
    nlp.Defaults.stop_words.add("t")
    return nlp
    
def apply_data(data, nlp):
    data['text']=data['text'].apply(lambda x: jrd.remove_stopwords(x, nlp))
    data['text']=data['text'].apply(lambda x: jrd.lemmatize(x, nlp))
    return data
    
def tokenizer():
    max_words = 5000
    
    tokenizer = Tokenizer(num_words=max_words)
    return tokenizer
    
def reviews(tokenizer, data):
    max_len = 500
    tokenizer.fit_on_texts(data.text)
    sequences = tokenizer.texts_to_sequences(data.text)
    reviews = pad_sequences(sequences, maxlen=max_len)
    return reviews
    
def labels(data):
    labels=to_categorical(data.label,num_classes=2)
    
def split_data(reviews):
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.1, stratify=labels)
    return X_train, X_test, y_train, y_test
     
def train_model(X_train, X_test, y_train, y_test):
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
    	             
def evaluate_model():
    names=['text', 'label']
    df_test = pd.read_csv('~/Documents/asi-project/data/test_dataset.csv', names=names)
    data_test=df_test[['text','label']]S
    clean_data(data_test, re_letters)
    data_test['text']=data_test['text'].apply(remove_stopwords)
    data_test['text']=data_test['text'].apply(lemmatize)
    
    new_sequences = tokenizer.texts_to_sequences(data_test.text)
    new_data = pad_sequences(new_sequences, maxlen=max_len)
    new_predictions = model3.predict(new_data)
    for i, prediction in enumerate(new_predictions):
        if prediction[1] > prediction[0]:
          label = 1
        else:
          label = 0

        if (data_test.label[i] == label):
          score = score + 1

    print(score)
    print(score / len(data_test))


def predict_model(data_test, user_input):
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_data = pad_sequences(user_sequence, maxlen=max_len)
    # Make predictions for the user input
    user_prediction = model3.predict(user_data)[0]
    # Print the predicted label for the user input
    if user_prediction[1] > user_prediction[0]:
        predicted_label = 1
    else:
        predicted_label = 0

    print("Predicted Label:", predicted_label)










