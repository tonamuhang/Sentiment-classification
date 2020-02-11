import nltk
#

import pandas as pd

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk import word_tokenize
from sklearn import model_selection, feature_extraction, preprocessing, svm, pipeline, metrics, tree, linear_model
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
import re
import csv

data = []
url = "tweets_train_retrieved_250000.csv"
cr = pd.read_csv(url)
print(cr.shape)
data_array = np.array(cr)

list_y = list(range(1379, 1412))
for i in range(0,155941):
  # print(1)
  if (data_array[i, 2] in list_y):
    data_array[i, 2] = data_array[i, 2]-1379
    data.append(data_array[i,:])
data = np.array(data)
print(data[0:5])
print(data.shape)
x_data = data[:, 1]
# x_data = x_data[1:]
y_data = data[:, 2]
y_data = list(map(int, y_data))
documents_categray = y_data
documents_categray = documents_categray
y_data = np.zeros((len(documents_categray), 33))
y_data[np.arange(len(documents_categray)),documents_categray] = 1

print(y_data.shape)
# y_data[-1] = 1389
x_data = x_data[0:10000]
y_data = y_data[0:10000]
# y_data = y_data[1:]
# unique_set = set(y_data[1:])
print(x_data[0:5])
print(y_data[-1])
# print(unique_set)/data/tweets_tokenized_60K.validate.txt", "Data source dev")

#############################  preprocessing 1#######################
#
# def clean_tweet(tw):
#     # tw = tw.strip()
#
#     # remove links
#     tw = re.sub(r'https?:\/\/.*[\r\n]*', '', str(tw))
#     # remove "pic."
#     tw = re.sub(r'pic.twitter.com\S+', '', str(tw))
#     tw = re.sub(r'@\S+', '', str(tw))
#     # remove "RT" tags
#     tw = re.sub('RT : ', '', tw)
#     return tw
# x_data = [clean_tweet(tw) for tw in x_data]
#
#
# from nltk import *
# from nltk.corpus import wordnet
# # nltk.download('punkt')
# nltk.download('wordnet')
# def replace(word):
#     repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)')
#     repl = r'\1\2\3'
#     # check if it is a standard word
#     if wordnet.synsets(word):
#         return word
#     repl_word = repeat_reg.sub(repl, word)
#     if repl_word != word:
#         return replace(repl_word)
#     else:
#         return repl_word
# def remove_repeat(text):
#     replacer = RepeatReplacer()
#     for row_i in range(text.shape[0]):
#         for word_j in range(len(text[row_i])):
#             text[row_i][word_j] = replacer.replace(text[row_i][word_j])
#     return text
#
# x_data = [replace(doc) for doc in x_data]
# # nltk.download('wordnet')
# def processing(input_str):
#     input_str1 = re.sub("[^a-zA-Z0-9]", " ", input_str)
#     # Lowercase
#     input_words = input_str1.lower().split()
#     # stem porter
#     # porter = PorterStemmer()
#     # input_words = [porter.stem(words) for words in input_words]
#     # stemer lancaster
# #     stemmer = LancasterStemmer()
# #     input_words = [stemmer.stem(words) for words in input_words]
#     # lemmatization of words
# #     lemmatizer = WordNetLemmatizer()
# #     input_words = [lemmatizer.lemmatize(words) for words in input_words]
#     output_str = (" ".join(input_words))
#     return output_str
#
# x_data = [processing(doc) for doc in x_data]    #perform general processing
# documents_text = [s.strip() for s in x_data]
# # Load and preprocess data
# sentences = documents_text
# print(x_data[0])
# print(y_data.shape)
######################  end   ###############################################





####################### preprocessing2 ################################


import numpy as np
import re
import itertools
from collections import Counter
def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


documents_text = [s.strip() for s in x_data]
# Load and preprocess data
sentences = documents_text
sentences = [clean_str(sent) for sent in sentences]

######################  end   ###############################################

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]







sentences = [s.split(" ") for s in sentences]
labels = y_data
sentences_padded = pad_sentences(sentences)
print(sentences_padded[0])
vocabulary, vocabulary_inv = build_vocab(sentences_padded)
x, y = build_input_data(sentences_padded, labels, vocabulary)
print(y.shape)
from keras import losses
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
print(y_train.shape)
# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = x.shape[1] # 56
print(sequence_length)
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 32
drop = 0.5

epochs = 100
batch_size = 256

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=33, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=0.001)

model.compile(optimizer=adam, loss= 'categorical_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))  # starts training
