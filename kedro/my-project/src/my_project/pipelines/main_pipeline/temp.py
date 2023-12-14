!pip install -U spacy
!python -m spacy download en_core_web_md
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import seaborn as sns
import spacy
import re
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as P
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
tqdm.pandas()

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import pad_sequences, to_categorical
import jrdpackage as jrd
names=['text', 'label']
df = pd.read_csv('drive/MyDrive/train_dataset.csv', names=names)
df.sample(5)
df.info()
df.groupby('text').nunique()
sns.countplot(x='label',data=df)
data=df[['text','label']]
re_letters=re.compile(r"[^a-zA-Z\s']")

jrd.clean_data(data, re_letters)
data.sample(5)
nlp = spacy.load('en_core_web_md',disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')
nlp.Defaults.stop_words.add("game")
nlp.Defaults.stop_words.add("play")
nlp.Defaults.stop_words.add("t")

data['text']=data['text'].apply(jrd.remove_stopwords)

data['text']=data['text'].progress_apply(jrd.lemmatize)

max_words = 5000
max_len = 500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data.text)
sequences = tokenizer.texts_to_sequences(data.text)
reviews = pad_sequences(sequences, maxlen=max_len)

labels=to_categorical(data.label,num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.1, stratify=labels)

model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model3.add(layers.Dense(2,activation='softmax'))

model3.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

history = model3.fit(X_train,
                     y_train,
                     epochs=10,
                     validation_data=(X_test, y_test))
names=['text', 'label']
df_test = pd.read_csv('drive/MyDrive/test_dataset.csv', names=names)
data_test=df_test[['text','label']]
clean_data(data_test, re_letters)
data_test['text']=data_test['text'].apply(remove_stopwords)
data_test['text']=data_test['text'].progress_apply(lemmatize)
data_test.sample(5)
sns.countplot(x='label',data=data_test)
# Make predictions on new text data using the pre-trained model
new_sequences = tokenizer.texts_to_sequences(data_test.text)
new_data = pad_sequences(new_sequences, maxlen=max_len)
new_predictions = model3.predict(new_data)

score = 0
print(len(data_test))

# Print the predicted labels for new text data
for i, prediction in enumerate(new_predictions):
    if prediction[1] > prediction[0]:
      label = 1
    else:
      label = 0

    if (data_test.label[i] == label):
      score = score + 1

print(score)
print(score / len(data_test))
