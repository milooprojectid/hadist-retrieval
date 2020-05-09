from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
import tqdm
import pandas as pd
import numpy as np

def load_stopwords(file):
    return [line.rstrip('\n\r') for line in open(file)]

def text_lower(text):
    return text.lower()

def remove_entities(text):
    return re.sub(r'\[[^]]*\]', '', text)

def case_folding(text):
    return re.sub(r'[^a-z]', ' ', re.sub("'", '', text))

def stemming(text):
    return stemmer.stem(text)

def stopwords_removal(texts, stopwords):
    texts_token = texts.split()
    not_stopword = []
    for token in texts_token:
        if token not in stopwords:
            not_stopword.append(token)
    return ' '.join(not_stopword)

def preprocessing(text, stopwords):
    tx_lower = text_lower(text)
    tx_remove_entities = remove_entities(tx_lower)
    tx_case_folding = case_folding(tx_remove_entities)
    tx_stemming = stemming(tx_case_folding)
    tx_stopword = stopwords_removal(tx_stemming, stopwords)
    return tx_stopword

stopwords = load_stopwords('./data_label/stopwords/stopword_list_TALA.txt')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

df_hadist_bukhari = pd.read_csv('./datasets/had_abudaud.csv',names=['L1','L2','Text','Processed'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_hadist_bukhari.Processed)
vectorizer.get_feature_names()[:10]

sentence = 'bagaimana cara bercerai ?'

sent_prep = preprocessing(sentence, stopwords)
query = sent_prep.split()
res = np.zeros(X.shape[0])
not_in_corpus = []

for keyword in query:
    try:
        res += X.toarray()[:,vectorizer.get_feature_names().index(keyword)]
    except:
        not_in_corpus.append(keyword)
        res = np.zeros(X.shape[0])

top_idx = np.argsort(-res)[:10]
valid = False
if sum(res)>0:
    valid = True

if valid:
    for i in range(len(top_idx)):
        print(res[top_idx[i]], df_hadist_bukhari.iloc[top_idx[i]][2] + "\n")
else:
    print('dindt match, someting wrong')
