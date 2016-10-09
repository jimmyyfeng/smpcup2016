import numpy as np
import pandas as pd

stopwords = open(r'/home/wym/data/smpcup/stopwords').readlines()
stopwords = [i.strip() for i in stopwords]

feat = eee.copy()
# feat1['recontent'] = [i for i in eee.content]
recontent = []


for cont in eee.content:
    splt = cont.strip().split(' ')
    lst = str()
    for word in splt:
        if word not in stopwords:
            lst += word
            lst += ' '
    recontent.append(lst)

feat['recontent'] = recontent

feat.to_csv(r'../feat.csv', index=0)

