import pandas as pd
import re


df = pd.read_csv('5822100/emoji_map_1791.csv')

emoji = df.iloc[:, 0]

mylist = [1381, 1392, 1389, 1403, 1397, 1384, 1380, 1394, 1393, 1399, 1391, 1388, 1400, 1395, 1396]


for j, e in enumerate(emoji):
    for i in mylist:
        if j == i:
            print(e, " "+ df.iloc[j, 2])