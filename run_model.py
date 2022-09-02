import numpy as np
import re
import pandas as pd
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#import seaborn as sb
from sklearn.metrics import classification_report, confusion_matrix
from config import MODELS_PATH

PROCESSED_DATA = MODELS_PATH / 'processed_data.bin'

glove_model = api.load('glove-twitter-25')
data = pd.DataFrame()
with open(PROCESSED_DATA) as f:
    data = f
def sentence_distance_from_word(good_or_bad, sentence):
    # we define sentence disance as the highest similarity of any of the words in the sentecnce with the target word.
    # will tokenizing - remover non-letters and lowercase the word
    list_of_word_vectors = [glove_model[item] for item in re.split('\W+', sentence.lower()) if
                            glove_model.__contains__(item)]

    if len(list_of_word_vectors) == 0:
        return 0

    # return the closest vector
    return max([cosine_similarity(X=[vector], Y=[glove_model[good_or_bad]])[0][0] for vector in list_of_word_vectors])

sentence_distance_from_word('good', 'be best')

df['distance_to_good'] = df['txt'].apply(lambda x : sentence_distance_from_word('good', x))
df['distance_to_bad'] = df['txt'].apply(lambda x : sentence_distance_from_word('bad', x))