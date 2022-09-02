import os
from pathlib import Path

DATA_FILE_TRAIN = Path.cwd().parent / 'data' / 'labeledTrainData.tsv'
CORPUS = Path.cwd().parent / 'data' / 'corpus.json'
SENTIMENTS = Path.cwd().parent / 'data' / 'sentiments.json'
MODELS_PATH = Path.cwd().parent / 'models'

MODEL = MODELS_PATH / 'raw_model'
TRAINED_MODEL = MODELS_PATH / 'trained_model'
PADDED_SENTENCES = MODELS_PATH / 'padded_sentences.npy'

VOCAB_LENGTH = 2470