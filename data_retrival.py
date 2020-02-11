import pandas as pd
import tweepy
import urllib3
from bs4 import BeautifulSoup
import numpy as np
import multiprocessing
import datetime

def retrieve_tweets(id):
    url = "https://twitter.com/i/web/status/" + str(id)

    response = http.request('GET', url)
    soup = BeautifulSoup(response.data.decode('utf-8'), "html.parser")
    content = soup.find("p", class_="TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text")
    if content != None:
        return content.text
    else:
        return np.NaN


# wrapper
def retrieve_mult(data):
    data['id'] = data['id'].apply(lambda x: retrieve_tweets(x))

    return data


# Share the dataframe object between processes
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# Tweepy set-up

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
#
# api = tweepy.API(auth)
#
# tweets = api.get_status(743285629576843264)

print(datetime.datetime.now())

http = urllib3.PoolManager()
text = pd.read_csv('5822100/full_train_plaintext.txt', sep = '\t')
emoji_map = pd.read_csv('5822100/emoji_map_1791.csv', sep = ',')
procs = []
# Retrieve tweet id's with only 1 emoji and restrict the emoji to emotion category
# To get the text description of the emoji use "emoji_map.iloc[1381].title"

text = text.query('~annotations.str.contains(",")', engine = 'python')
text['annotations'] = text['annotations'].astype(int)
text = text.query('1379 <= annotations <= 1411', engine= 'python')

tweet = text['id']
emoji = text['annotations']

print(tweet.head(5))
print(emoji.head(5))
print(len(text))


# for i in range(100000):
#     result = retrieve_tweets(tweet.iloc[i])
#     text['id'].iloc[i] = result
#     print("Process: ", i/100000)

text = parallelize_dataframe(text, retrieve_mult)

text = text.dropna(axis=0, how='any')
print(len(text))

retrieved = text[['id', 'annotations']].to_csv('tweets_train_retrieved.csv')
print(datetime.datetime.now())
