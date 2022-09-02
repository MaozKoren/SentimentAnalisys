import numpy as np
import csv
import pickle
import sys
import re
import json
import pandas as pd
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#import seaborn as sb
from sklearn.metrics import classification_report, confusion_matrix
from config import DATA_FILE_TRAIN, MODELS_PATH, CORPUS, SENTIMENTS
from numpy import array

PROCESSED_DATA = MODELS_PATH / 'processed_data.bin'

# df = pd.read_csv(DATA_FILE_TRAIN, header=None, delimiter='\t', names=['txt', 'sentiment'])

corpus = []
sentiments = []

def preprocess_data():
    with open(DATA_FILE_TRAIN, encoding='utf8') as f,\
            open(CORPUS, 'w') as c,\
            open(SENTIMENTS, 'w') as s:

        next(f)

        for line in f:
            _, sentiment, review = line.split('\t')
            corpus.append(review)
            sentiments.append(int(sentiment))

        json.dump(sentiments, s)
        json.dump(corpus, c)


def load_corpus():
    with open(CORPUS) as f:
        return json.load(f)

def load_sentiments():
    with open(SENTIMENTS) as f:
        return array(json.load(f))