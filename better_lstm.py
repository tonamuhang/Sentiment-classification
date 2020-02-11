import multiprocessing
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import utils
from sklearn.utils import class_weight
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
from sklearn.utils import shuffle

# DIM = 128
# batch_size = 32
embedding_size = 100
vocabulary_size = 10000
max_len = 100   # Largest # of words used(Tweet limits in 250 characters)


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
    tweet = shuffle(tweet)
    X = tweet['id']
    y = tweet['annotations']
    return X, y


def map_to_keywords(index):
    string = emoji_map['keywords'][index]
    return string[1:-1].replace(",", "")


# X, y = load_data('train.csv')
# test_X, test_y = load_data('test.csv')
# emoji_map = pd.read_csv('5822100/emoji_map_1791.csv')
X, y = load_data('tweets_train_retrieved.csv')
test_X, test_y = load_data('tweets_test_retrieved.csv')
emoji_map = pd.read_csv('5822100/emoji_map_1791.csv')

X = X[0:50000]
y = y[0:50000]

test_X = test_X[0:1000]
test_y = test_y[0:1000]

# for i in range(1379, 1411):
#   print(i, ": ", list(y).count(i) / len(y))

# Map the emoji id to its keywords
y = y.apply(lambda x : map_to_keywords(x))
test_y = test_y.apply(lambda x : map_to_keywords(x))
# target_y = target_y.apply(lambda x : map_to_keywords(x))

label_encoder = LabelEncoder()
lable_encoder = label_encoder.fit(y)
print(label_encoder.classes_)
integer_encoded = label_encoder.transform(y)
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
target_y = onehot_encoder.fit_transform(integer_encoded)


# inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[2, :])])
# print(inverted)

integer_encoded = label_encoder.fit_transform(test_y)
test_y = to_categorical(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


print("### Training word model")
word_model = gensim.models.Word2Vec(sentences=X, size=embedding_size, min_count=1, iter=1)
target_word_model = gensim.models.Word2Vec(sentences=y, size=embedding_size, min_count=0, iter=1)
embedding_matrix_X = np.zeros((len(word_model.wv.vocab) + 1, embedding_size))
embedding_matrix_y = np.zeros((len(target_word_model.wv.vocab) + 1, embedding_size))

for i, vec in enumerate(word_model.wv.vectors):
    embedding_matrix_X[i] = vec
for i, vec in enumerate(target_word_model.wv.vectors):
    embedding_matrix_y[i] = vec

print("### Done!")

# Tokenize X
tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
tokenizer = Tokenizer(num_words= embedding_size)

X = X.apply(lambda x: tknzr.tokenize(x))
test_X = test_X.apply(lambda x: tknzr.tokenize(x))
y = y.apply(lambda x: tknzr.tokenize(x))

tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

tokenizer.fit_on_texts(test_X)
sequences = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(sequences, maxlen=max_len, padding='post')

tokenizer.fit_on_texts(y)
sequences = tokenizer.texts_to_sequences(y)
y = pad_sequences(sequences, maxlen=max_len, padding='post')









# Average sentence embedding for X
left = Sequential()
left.add(Embedding(input_dim=len(word_model.wv.vocab)+1, output_dim=embedding_size, input_length=X.shape[1], weights=[embedding_matrix_X], trainable=False))
left.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))

# Average sentence embedding for y
right = Sequential()
right.add(Embedding(input_dim=len(target_word_model.wv.vocab)+1,output_dim=embedding_size, input_length=y.shape[1], weights=[embedding_matrix_y], trainable=False))
# right.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))

# Merged layer
merged = keras.layers.Dot(axes=-1, normalize=True)([left.output, right.output])
merged = keras.layers.Lambda(lambda x: keras.backend.expand_dims(merged, axis=-1))(merged)
merged = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(merged)
# merged = Dense(100, activation='relu')(merged)
# merged = Dropout(0.5)(merged)
merged = Dense(33, activation='softmax')(merged)

model = keras.models.Model([left.input, right.input], merged)

model.summary()

model.compile(optimizer='sgd', loss= 'categorical_crossentropy', metrics=['accuracy'])
model.fit([X, y], target_y, validation_split=0.2, epochs=3)