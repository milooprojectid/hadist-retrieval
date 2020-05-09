from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
import tqdm
import pandas as pd
import numpy as np

def load_stopwords(file):
    return [line.rstrip('\n\r') for line in open(file)]

stopwords = load_stopwords('./data_label/stopwords/stopword_list_TALA.txt')

def text_lower(text):
    return text.lower()

def remove_entities(text):
    return re.sub(r'\[[^]]*\]', '', text)

def case_folding(text):
    return re.sub(r'[^a-z]', ' ', re.sub("'", '', text))

def stemming(text):
    return stemmer.stem(text)

def stopwords_removal(texts):
    texts_token = texts.split()
    not_stopword = []
    for token in texts_token:
        if token not in stopwords:
            not_stopword.append(token)
    return ' '.join(not_stopword)

def preprocessing(text):
    tx_lower = text_lower(text)
    tx_remove_entities = remove_entities(tx_lower)
    tx_case_folding = case_folding(tx_remove_entities)
    tx_stemming = stemming(tx_case_folding)
    tx_stopword = stopwords_removal(tx_stemming)
    return tx_stopword

stopwords = load_stopwords('./data_label/stopwords/stopword_list_TALA.txt')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

hadist = pd.read_csv('./datasets/hadist.csv', delimiter=';')

preprocessed = hadist.Text.apply(preprocessing)
preprocessed.to_csv('../datasets/hadist_processed.csv')
