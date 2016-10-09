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

#抽取训练集词典
doc = [i.strip().split(' ') for i in train.recontent]
words = []
for i in doc:
    words.extend(i)
wordset = set(words)

#统计训练集男/女性微博高频词
fdoc = [i.strip().split(' ') for i in train[train.sex == 'f'].recontent]
fwords = []
for i in fdoc:
    fwords.extend(i)
mdoc = [i.strip().split(' ') for i in train[train.sex == 'm'].recontent]
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
ffre.columns = [['word', 'fcount']]
mfre = pd.DataFrame.from_dict(mwordcount)
mfre.columns = ['word', 'mcount']
fre = pd.merge(ffre, mfre, on='word')
fre['frate'] = fre.fcount / (fre.fcount+fre.mcount)
fre['mrate'] = fre.mcount / (fre.fcount+fre.mcount)
topfwords = fre[(fre.fcount>40)&(fre.frate>=0.45)].sort_values(by='frate', ascending=0)
topfwords.to_csv(r'../topfwords.csv', index=0)
topmwords = fre[(fre.mcount>150)&(fre.mrate>=0.8)].sort_values(by='mrate', ascending=0)
topmwords.to_csv(r'../topmwords.csv', index=0)


#统计训练集各年龄段微博高频词
olddoc = [i.strip().split(' ') for i in train[train.age2 == '-1979'].recontent]
oldwords = []
for i in olddoc:
    oldwords.extend(i)

middoc = [i.strip().split(' ') for i in train[train.age2 == '1980-1989'].recontent]
midwords = []
for i in middoc:
    midwords.extend(i)

youngdoc = [i.strip().split(' ') for i in train[train.age2 == '1990+'].recontent]
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
oldfre.columns = ['word', 'oldcount']
midfre = pd.DataFrame.from_dict(midwordcount)
midfre.columns = ['word', 'midcount']
youngfre = pd.DataFrame.from_dict(youngwordcount)
youngfre.columns = ['word', 'youngcount']

fre = pd.merge(oldfre, midfre, on='word')
fre = pd.merge(fre, youngfre, on='word')

fre['oldrank'] = fre.oldcount.rank(method='max')
fre['midrank'] = fre.midcount.rank(method='max')
fre['youngrank'] = fre.youngcount.rank(method='max')

fre['oldrate'] = fre.oldcount / (fre.oldcount+fre.midcount+fre.youngcount)
fre['midrate'] = fre.midcount / (fre.oldcount+fre.midcount+fre.youngcount)
fre['youngrate'] = fre.youngcount / (fre.oldcount+fre.midcount+fre.youngcount)

fre[(fre.oldcount> 80)&(fre.oldrate>=0.6)].sort_values(by='oldrate', ascending=0).to_csv(r'../topoldwordcount.csv', index=False)
fre[(fre.midcount> 100)&(fre.midrate>=0.8)].sort_values(by='midrate', ascending=0).to_csv(r'../topmidwordcount.csv', index=False)
fre[(fre.youngcount> 50)&(fre.youngrate>=0.7)].sort_values(by='youngrate', ascending=0).to_csv(r'../topyoungwordcount.csv', index=False)


#统计训练集各地区微博高频词
dbdoc = [i.strip().split(' ') for i in train[train.loction2 == '东北'].recontent]
dbwords = []
for i in dbdoc:
    dbwords.extend(i)
    
    
hbdoc = [i.strip().split(' ') for i in train[train.loction2 == '华北'].recontent]
hbwords = []
for i in hbdoc:
    hbwords.extend(i)
    

hddoc = [i.strip().split(' ') for i in train[train.loction2 == '华东'].recontent]
hdwords = []
for i in hddoc:
    hdwords.extend(i)
    
hzdoc = [i.strip().split(' ') for i in train[train.loction2 == '华中'].recontent]
hzwords = []
for i in hzdoc:
    hzwords.extend(i)
    
hndoc = [i.strip().split(' ') for i in train[train.loction2 == '华南'].recontent]
hnwords = []
for i in hndoc:
    hnwords.extend(i)
    
xndoc = [i.strip().split(' ') for i in train[train.loction2 == '西南'].recontent]
xnwords = []
for i in xndoc:
    xnwords.extend(i)
    
xbdoc = [i.strip().split(' ') for i in train[train.loction2 == '西北'].recontent]
xbwords = []
for i in xbdoc:
    xbwords.extend(i)
    
jwdoc = [i.strip().split(' ') for i in train[train.loction2 == '海外'].recontent]
jwwords = []
for i in jwdoc:
    jwwords.extend(i)

dbwordcount = {}
hbwordcount = {}
hdwordcount = {}
hzwordcount = {}
hnwordcount = {}
xnwordcount = {}
xbwordcount = {}
jwwordcount = {}

for item in wordset:
    dbwordcount[item] = dbwords.count(item)
for item in wordset:
    hbwordcount[item] = hbwords.count(item)
for item in wordset:
    hdwordcount[item] = hdwords.count(item)
for item in wordset:
    hzwordcount[item] = hzwords.count(item)
for item in wordset:
    hnwordcount[item] = hnwords.count(item)
for item in wordset:
    xnwordcount[item] = xnwords.count(item)
for item in wordset:
    xbwordcount[item] = xbwords.count(item)
for item in wordset:
    jwwordcount[item] = jwwords.count(item)

dbwordcount = sorted(dbwordcount.iteritems(), key=lambda d:d[1], reverse = True)
hbwordcount = sorted(hbwordcount.iteritems(), key=lambda d:d[1], reverse = True)
hdwordcount = sorted(hdwordcount.iteritems(), key=lambda d:d[1], reverse = True)
hzwordcount = sorted(hzwordcount.iteritems(), key=lambda d:d[1], reverse = True)
hnwordcount = sorted(hnwordcount.iteritems(), key=lambda d:d[1], reverse = True)
xnwordcount = sorted(xnwordcount.iteritems(), key=lambda d:d[1], reverse = True)
xbwordcount = sorted(xbwordcount.iteritems(), key=lambda d:d[1], reverse = True)
jwwordcount = sorted(jwwordcount.iteritems(), key=lambda d:d[1], reverse = True)

dbfre = pd.DataFrame.from_dict(dbwordcount)
dbfre.columns = ['word', 'dbcount']
hbfre = pd.DataFrame.from_dict(hbwordcount)
hbfre.columns = ['word', 'hbcount']
hdfre = pd.DataFrame.from_dict(hdwordcount)
hdfre.columns = ['word', 'hdcount']
hzfre = pd.DataFrame.from_dict(hzwordcount)
hzfre.columns = ['word', 'hzcount']
hnfre = pd.DataFrame.from_dict(hnwordcount)
hnfre.columns = ['word', 'hncount']
xnfre = pd.DataFrame.from_dict(xnwordcount)
xnfre.columns = ['word', 'xncount']
xbfre = pd.DataFrame.from_dict(xbwordcount)
xbfre.columns = ['word', 'xbcount']
jwfre = pd.DataFrame.from_dict(jwwordcount)
jwfre.columns = ['word', 'jwcount']

locfre = pd.merge(dbfre, hbfre, on='word')
locfre = pd.merge(locfre, hdfre, on='word')
locfre = pd.merge(locfre, hzfre, on='word')
locfre = pd.merge(locfre, hnfre, on='word')
locfre = pd.merge(locfre, xnfre, on='word')
locfre = pd.merge(locfre, xbfre, on='word')
locfre = pd.merge(locfre, jwfre, on='word')


locfre['dbrate']=locfre.dbcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['hbrate']=locfre.hbcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['hdrate']=locfre.hdcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['hzrate']=locfre.hzcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['hnrate']=locfre.hncount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['xnrate']=locfre.xncount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['xbrate']=locfre.xbcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)
locfre['jwrate']=locfre.jwcount/(locfre.dbcount+locfre.hbcount+locfre.hdcount+locfre.hzcount+locfre.hncount+locfre.xncount+locfre.xbcount+locfre.jwcount)

locfre[(locfre.dbcount> 10)&(locfre.dbrate>=0.8)].sort_values(by='dbrate', 
                                            ascending=0).to_csv(r'../topdbword.csv', index=0)
locfre[(locfre.hdcount> 50)&(locfre.hdrate>=0.9)].sort_values(by='hdrate', 
                                            ascending=0).to_csv(r'../tophdword.csv', index=0)    
locfre[(locfre.hbcount> 100)&(locfre.hbrate>=0.8)].sort_values(by='hbrate', 
                                                ascending=0).to_csv(r'../tophbword.csv', index=0)
locfre[(locfre.hzcount> 20)&(locfre.hzrate>=0.9)].sort_values(by='hzrate', 
                                            ascending=0).to_csv(r'../tophzword.csv', index=0)
locfre[(locfre.hncount> 50)&(locfre.hnrate>=0.8)].sort_values(by='hnrate', 
                                            ascending=0).to_csv(r'../tophnword.csv', index=0)
locfre[(locfre.xbcount> 50)&(locfre.xbrate>=0.8)].sort_values(by='xbrate', 
                                        ascending=0).to_csv(r'../topxbword.csv', index=0)
locfre[(locfre.xncount> 50)&(locfre.xnrate>=0.8)].sort_values(by='xnrate', 
                                            ascending=0).to_csv(r'../topxnword.csv', index=0)
locfre[(locfre.jwcount>= 5)&(locfre.jwrate>=0.9)].sort_values(by='jwrate', 
                                        ascending=0).to_csv(r'../topjwword.csv', index=0)
