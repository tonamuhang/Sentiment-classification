import multiprocessing
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

embedding_size = 100
max_len = 100   # Largest # of words used(Tweet limits in 250 characters)


def build_model(class_list):
    models = []
    for i, c in enumerate(class_list):
        models.append({})
        models[i]['input'] = keras.layers.Input(shape=(20, ))
        models[i]['embedding'] = Embedding(len(target_word_model.wv.vocab)+1,
                                           embedding_size, input_length=(20, ),
                                           weights=[embedding_matrix_y],
                                           trainable=False)(models[i]['input'])
        models[i]['lstm'] = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(models[i]['embedding'])
        models[i]['dense'] = Dense(30, activation='relu')(models[i]['lstm'])
        models[i]['dropout'] = Dropout(0.3)(models[i]['dense'])
        models[i]['model'] = keras.models.Model(input=[models[i]['input']],
                                                 output=[models[i]['dropout']])

    return models


def load_data(filename):
    tweet = pd.read_csv(filename, lineterminator='\n')
    tweet = utils.shuffle(tweet)
    X = tweet['id']
    y = tweet['annotations']
    return X, y


def map_to_keywords(index):
    string = emoji_map['keywords'][index]
    return string[1:-1].replace(",", "")

X, y = load_data('train.csv')
test_X, test_y = load_data('test.csv')
emoji_map = pd.read_csv('5822100/emoji_map_1791.csv')

# X = X[0:50000]
# y = y[0:50000]
#
# test_X = test_X[0:1000]
# test_y = test_y[0:1000]


label_encoder = LabelEncoder()
lable_encoder = label_encoder.fit(y)


y_class = label_encoder.classes_
y_class = [map_to_keywords(x) for x in y_class]
y_class = np.asarray(y_class)

integer_encoded = label_encoder.transform(y)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
target_y = onehot_encoder.fit_transform(integer_encoded)

print("### Training word model")
word_model = gensim.models.Word2Vec(X, size=embedding_size, min_count=1, iter=1)
target_word_model = gensim.models.Word2Vec(y_class, size=embedding_size, min_count=0, iter=1)
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
y_class = [tknzr.tokenize(x) for x in y_class]

tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
tokenizer.fit_on_texts(test_X)
sequences = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(sequences, maxlen=max_len, padding='post')

tokenizer = Tokenizer(num_words= len(target_word_model.wv.vocab)+1)
tokenizer.fit_on_texts(y_class)
sequences = tokenizer.texts_to_sequences(y_class)
y_class = pad_sequences(sequences, maxlen=20, padding='post')
# y_class = tokenizer.texts_to_sequences(y_class)


# Define model
models = build_model(y_class)
merged_layers = []

X_inputs = keras.layers.Input(shape=(X.shape[1],))
embedding = Embedding(len(word_model.wv.vocab)+1,
                      embedding_size, input_length=X.shape[1],
                      weights=[embedding_matrix_X], trainable=False)(X_inputs)
bilstm = keras.layers.Bidirectional(LSTM(15,
                                         dropout=0.2, recurrent_dropout=0.2))(embedding)


for i in range(len(models)):
    merged_layers.append(keras.layers.Dot(axes=-1, normalize=True)([bilstm, models[i]['model'].output]))

merged = keras.layers.Concatenate(axis=-1)([merged_layers[0], merged_layers[1]])
for i in range(2, len(merged_layers)):
    merged = keras.layers.Concatenate(axis=-1)([merged, merged_layers[i]])


dropout_1 = Dropout(0.4)(merged)
prediction = Dense(15, activation='softmax')(dropout_1)
inputs = [X_inputs]
for m in models:
    inputs.append(m['input'])

model = keras.models.Model(inputs, prediction)

# model.summary()
model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])


inputs = [X]

y_input = []
for i in range(len(X)):
    temp = []
    for j in range(len(y_class)):
        temp.append(y_class[j])
    y_input.append(temp)



# y_input = np.asarray(y_class)
y_input = np.asarray(y_input)
y_input = np.swapaxes(y_input, 0, 1)
print(y_input.shape)


for i in range(15):
    inputs.append(y_input[i])

# print(inputs)
model.fit(inputs, target_y, epochs=3, validation_split=0.2)



