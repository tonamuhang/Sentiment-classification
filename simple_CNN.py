import multiprocessing
from sklearn import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.utils import to_categorical
from keras.constraints import min_max_norm
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

def load_data(filename):
    tweet = pd.read_csv(filename, lineterminator='\n')
    X = tweet['id']
    y = tweet['annotations']
    return X, y

def get_tagged_doc(filename):
    data = pd.read_csv(filename, lineterminator='\n')
    tagged = data.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['id']), tags=[r.annotations]), axis=1)
    return tagged


DIM = 128
batch_size = 32
embedding_size = 100
vocabulary_size = 25000
max_len = 100   # Largest # of words used(Tweet limits in 250 characters)

# Simple RNN with sentence embedding

history = load_model("models/simplernn")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')

plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# TODO: 1. Implement sentence embedding. 2. Implement CNN