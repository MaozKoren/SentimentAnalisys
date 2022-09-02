import pathlib
import pickle

import data_processing as dp
import numpy
from config import VOCAB_LENGTH, MODEL, TRAINED_MODEL, PADDED_SENTENCES

from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Embedding



def create_paddings(corpus, path_to_file):

    all_words = []
    print('going to tokenize')
    for sent in corpus:
        tokenize_word = word_tokenize(sent)
        for word in tokenize_word:
            all_words.append(word)
    print('finished to tokenize')

    unique_words = set(all_words)

    embedded_sentences = [one_hot(sent, VOCAB_LENGTH) for sent in corpus]

    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(corpus, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))

    print('going to pad_sequences')

    padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
    numpy.save(path_to_file, padded_sentences)

    print('finished to pad_sequences')


def create_model():

    corpus = dp.load_corpus()
    create_paddings(corpus, PADDED_SENTENCES)
    padded_sentences = numpy.load(PADDED_SENTENCES)

    print('going to add Embedding')
    model = Sequential()
    model.add(Embedding(VOCAB_LENGTH, 20, input_length=length_long_sentence))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print('finished to add Embedding')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.save(MODEL)

    return model


if not pathlib.Path(MODEL).is_dir():
    print('going to create model')
    model = create_model()
else:
    print('loading existing model')
    model = load_model(MODEL)


def train_model(epcohs):
    sentiments = dp.load_sentiments()

    print('going to load padded sentences')
    padded_sentences = numpy.load(PADDED_SENTENCES)

    print(f'going to fit model with {epcohs} epochs')
    model.fit(padded_sentences, sentiments, epochs=epcohs, verbose=1)
    model.save(TRAINED_MODEL)
    model.predict()
    return model

def predict(model):
    model.pr

EPOCHS = 12

model = train_model(EPOCHS)
# loss, accuracy = model.evaluate(padded_sentences, sentiments, verbose=0)

print('Accuracy: %f' % (accuracy * 100))
