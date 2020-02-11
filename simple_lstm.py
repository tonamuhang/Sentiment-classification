import pandas as pd
import numpy as np
from gensim.test.utils import common_texts
from keras.utils import to_categorical
from keras.models import Sequential
from keras.constraints import min_max_norm
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import data_preprocess
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DIM = 128
batch_size = 32
embedding_size = 100

vocabulary_size = 20000
max_len = 100   # Longest # of words used(Tweet limits in 250 characters)


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

vect = TfidfVectorizer( ngram_range=(1,1), stop_words=None, min_df = 0,binary=True).fit(X)
X = vect.transform(X)

label_encoder = LabelEncoder()
lable_encoder = label_encoder.fit(y)
integer_encoded = label_encoder.transform(y)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
target_y = onehot_encoder.fit_transform(integer_encoded)


model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_size, input_length=X.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(33, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(X, target_y, validation_split=0.2, epochs=10)

