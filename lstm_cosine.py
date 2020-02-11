import multiprocessing
import pandas as pd
import re

from sklearn import utils
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.constraints import min_max_norm
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

DIM = 128
batch_size = 32
embedding_size = 100
vocabulary_size = 25000
max_len = 100   # Largest # of words used(Tweet limits in 250 characters)

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def load_data(filename):
    tweet = pd.read_csv(filename, lineterminator='\n')
    X = tweet['id']
    y = tweet['annotations']
    return X, y


def map_to_keywords(index):
    string = emoji_map['keywords'][index]
    return string[1:-1].replace(",", "")


X, y = load_data('tweets_train_retrieved_250000.csv')
test_X, test_y = load_data('tweets_test_retrieved100000.csv')
emoji_map = pd.read_csv('5822100/emoji_map_1791.csv')

# Map the emoji id to its keywords
y = y.apply(lambda x : map_to_keywords(x))
test_y = test_y.apply(lambda x : map_to_keywords(x))


print("### Training word model")
word_model = gensim.models.Word2Vec(X, size=embedding_size, min_count=1, iter=1)
target_word_model = gensim.models.Word2Vec(y, size=embedding_size, min_count=0, iter=1)
embedding_matrix_X = np.zeros((len(word_model.wv.vocab) + 1, embedding_size))
embedding_matrix_y = np.zeros((len(target_word_model.wv.vocab)+1, embedding_size))

for i, vec in enumerate(word_model.wv.vectors):
    embedding_matrix_X[i] = vec
for i, vec in enumerate(target_word_model.wv.vectors):
    embedding_matrix_y[i] = vec

print("### Done!")

# Tokenize X
tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
tokenizer = Tokenizer(num_words= len(word_model.wv.vocab)+1)

X = X.apply(lambda x: tknzr.tokenize(x))
test_X = test_X.apply(lambda x: tknzr.tokenize(x))
y = y.apply(lambda x: tknzr.tokenize(x))

tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=max_len)
tokenizer.fit_on_texts(test_X)
sequences = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(sequences, maxlen=max_len)
tokenizer.fit_on_texts(y)
sequences = tokenizer.texts_to_sequences(y)
y = pad_sequences(sequences, maxlen=max_len)

# Average sentence embedding for X
left = Sequential()
left.add(Embedding(len(word_model.wv.vocab)+1, embedding_size, input_length=X.shape[1], weights=[embedding_matrix_X], trainable=False))
# left.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
# Bi-LSTM layers
left.add(keras.layers.Bidirectional(LSTM(64)))
left.add(keras.layers.GlobalMaxPooling1D())
left.add(Dense(100, activation='relu'))
# Average sentence embedding for y
right = Sequential()
right.add(Embedding(len(target_word_model.wv.vocab)+1, embedding_size, input_length=y.shape[1], weights=[embedding_matrix_y], trainable=False))
# right.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
right.add(keras.layers.Bidirectional(LSTM(64)))
right.add(keras.layers.MaxPooling1D())
right.add(Dense(100, activation='relu'))

# Merged layer
merged = keras.layers.Dot(axes=-1, normalize=True)([left.output, right.output])
merged = Dense(33, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(33, activation='softmax')(merged)

model = keras.models.Model([left.input, right.input], merged)

model.summary()
