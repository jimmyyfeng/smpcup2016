import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
import operator
from matplotlib import pylab as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

feat = pd.read_csv(r'../feat.csv')
train = feat.iloc[:3200, :]

doc = [i.strip().split(' ') for i in train.source]
words = []
for i in doc:
    words.extend(i)
wordset = set(words)

#统计训练集男/女性微博高频source
fdoc = [i.strip().split(' ') for i in train[train.sex == 'f'].source]
fwords = []
for i in fdoc:
    fwords.extend(i)
mdoc = [i.strip().split(' ') for i in train[train.sex == 'm'].source]
mwords = []
for i in mdoc:
    mwords.extend(i)
fwordcount = {}
mwordcount = {}
for item in wordset:
    fwordcount[item] = fwords.count(item)
for item in wordset:
    mwordcount[item] = mwords.count(item)
fwordcount = sorted(fwordcount.iteritems(), key=lambda d:d[1], reverse = True)    
mwordcount = sorted(mwordcount.iteritems(), key=lambda d:d[1], reverse = True)
ffre = pd.DataFrame.from_dict(fwordcount)
ffre.columns = [['source', 'fcount']]
mfre = pd.DataFrame.from_dict(mwordcount)
mfre.columns = ['source', 'mcount']
fre = pd.merge(ffre, mfre, on='source')
fre['frate'] = fre.fcount / (fre.fcount+fre.mcount)
fre['mrate'] = fre.mcount / (fre.fcount+fre.mcount)
topfwords = fre[(fre.fcount>30)&(fre.frate>=0.55)].sort_values(by='frate', ascending=0)
topfwords.to_csv(r'../topfsource.csv', index=0)
topmwords = fre[(fre.mcount>200)&(fre.mrate>=0.8)].sort_values(by='mrate', ascending=0)
topmwords.to_csv(r'../topmsource.csv', index=0)


#统计训练集各年龄段微博高频source
olddoc = [i.strip().split(' ') for i in train[train.age2 == '-1979'].source]
oldwords = []
for i in olddoc:
    oldwords.extend(i)

middoc = [i.strip().split(' ') for i in train[train.age2 == '1980-1989'].source]
midwords = []
for i in middoc:
    midwords.extend(i)

youngdoc = [i.strip().split(' ') for i in train[train.age2 == '1990+'].source]
youngwords = []
for i in youngdoc:
    youngwords.extend(i)

oldwordcount = {}
midwordcount = {}
youngwordcount = {}

for item in wordset:
    oldwordcount[item] = oldwords.count(item)
for item in wordset:
    midwordcount[item] = midwords.count(item)
for item in wordset:
    youngwordcount[item] = youngwords.count(item)

oldwordcount = sorted(oldwordcount.iteritems(), key=lambda d:d[1], reverse = True)
midwordcount = sorted(midwordcount.iteritems(), key=lambda d:d[1], reverse = True)
youngwordcount = sorted(youngwordcount.iteritems(), key=lambda d:d[1], reverse = True)

oldfre = pd.DataFrame.from_dict(oldwordcount)
oldfre.columns = ['source', 'oldcount']
midfre = pd.DataFrame.from_dict(midwordcount)
midfre.columns = ['source', 'midcount']
youngfre = pd.DataFrame.from_dict(youngwordcount)
youngfre.columns = ['source', 'youngcount']

fre = pd.merge(oldfre, midfre, on='source')
fre = pd.merge(fre, youngfre, on='source')

fre['oldrank'] = fre.oldcount.rank(method='max')
fre['midrank'] = fre.midcount.rank(method='max')
fre['youngrank'] = fre.youngcount.rank(method='max')

fre['oldrate'] = fre.oldcount / (fre.oldcount+fre.midcount+fre.youngcount)
fre['midrate'] = fre.midcount / (fre.oldcount+fre.midcount+fre.youngcount)
fre['youngrate'] = fre.youngcount / (fre.oldcount+fre.midcount+fre.youngcount)

fre[(fre.oldcount> 20)&(fre.oldrate>=0.8)].sort_values(by='oldrate', ascending=0).to_csv(r'../topoldsource.csv', index=False)
fre[(fre.midcount> 50)&(fre.midrate>=0.8)].sort_values(by='midrate', ascending=0).to_csv(r'../topmidsource.csv', index=False)
fre[(fre.youngcount> 50)&(fre.youngrate>=0.7)].sort_values(by='youngrate', ascending=0).to_csv(r'../topyoungsource.csv', index=False)
