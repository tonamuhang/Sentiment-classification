import pandas as pd
import keras
from keras.utils import to_categorical


'''
Input: file csv name
Output: X, y(tweet, emoji_id)
'''
def load_data(filename):
    tweet = pd.read_csv(filename, lineterminator='\n')
    X = tweet['id']
    y = tweet['annotations']
    return X, y

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded