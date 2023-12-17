import spacy
import re
import keras
import jrdpackage as jrd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

max_words = 5000
max_len = 500
names=['text', 'label']
re_letters=re.compile(r"[^a-zA-Z\s']")

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
    
def tokenizer():
    tokenizer = Tokenizer(num_words=max_words)
    return tokenizer

def prepare_data(df, nlp, tokenizer):
    data=df[['text','label']]
    jrd.clean_data(data, re_letters)

    data['text']=data['text'].apply(lambda x: jrd.remove_stopwords(x, nlp))
    data['text']=data['text'].apply(lambda x: jrd.lemmatize(x, nlp))
    
    labels=to_categorical(data.label,num_classes=2)

    tokenizer.fit_on_texts(data.text)
    sequences = tokenizer.texts_to_sequences(data.text)
    reviews = pad_sequences(sequences, maxlen=max_len)
    return labels, reviews
