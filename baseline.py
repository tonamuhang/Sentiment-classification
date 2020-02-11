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
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def load_data(filename):
    tweet = pd.read_csv(filename, lineterminator='\n')
    X = tweet['id']
    y = tweet['annotations']
    return X, y


X, y = load_data('train.csv')
test_X, test_y = load_data('test.csv')
# emoji_map = pd.read_csv('5822100/emoji_map_1791.csv')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, train_size=0.8)

X_train = X_train[0:50000]
y_train = y_train[0:50000]
X_test = X_test[0:50000]
y_test = y_test[0:50000]

# y = y.apply(lambda x: x - 1379)
# test_y = test_y.apply(lambda x: x - 1379)

# # y = np.array(y)
# y_binary = to_categorical(y)
# test_y_binary = to_categorical(test_y)
# # y = y.apply(lambda x: tknzr.tokenize(str(x)))

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

pipeline_LinearSVC = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(sublinear_tf=True,norm='l1')),
                     ('norm', Normalizer()),
                     ('clf', LinearSVC(C=0.215,max_iter=5000,penalty='l2'))
                    ])
parameters_LinearSVC = {}  # ignore best parameters search
grid_LinearSVC = GridSearchCV(pipeline_LinearSVC, param_grid=parameters_LinearSVC, cv=5, n_jobs=-1, verbose=3)

print("----------------------------------------------------------------------")
print("Linear SVC")
grid_LinearSVC.fit(X_train, y_train)
print ("score = %3.4f" %(grid_LinearSVC.score(X_test, y_test)))
print (grid_LinearSVC.best_params_)
prediction = grid_LinearSVC.predict(X_test)
print(classification_report(y_test, prediction))
cm = confusion_matrix(y_test, prediction)