import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
    

# ---read data---


merge_data = pd.read_csv('./data/newfeat/merge_data.csv')
newfeat = pd.read_csv('./data/newfeat/newmerge_feat.csv')
feat_time_3hour = pd.read_csv('./data/newfeat/feat_time_3hour.csv')
feat_time_1hour = pd.read_csv('./data/newfeat/feat_time_1hour.csv')
age_count = pd.read_csv('./data/newfeat/feat_age.csv')
gender_count = pd.read_csv('./data/newfeat/feat_sex.csv')
area_count = pd.read_csv('./data/newfeat/feat_area_word.csv')
area_count_word = pd.read_csv('./data/newfeat/feat_area.csv')


# ---split data---
newfeat = pd.merge(newfeat, feat_time_3hour, on = 'uid')
newfeat = pd.merge(newfeat, feat_time_1hour, on = 'uid')
age_sum_count = age_count
gender_sum_count = gender_count
area_sum_count = area_count

# del newfeat['name']
newfeat.fillna(0, inplace=1)

# newfeat = pd.concat([newfeat, probfeat], axis=1)
# age_feat = pd.concat([newfeat, age_count], axis=1)
# gender_feat = pd.concat([newfeat, gender_count], axis=1)

age_feat = pd.concat([newfeat, age_sum_count], axis=1)
gender_feat = pd.concat([newfeat, gender_sum_count], axis=1)
area_feat = pd.concat([newfeat, area_sum_count], axis=1)
area_feat = pd.concat([area_feat, area_count_word], axis=1)

train_age = age_feat.iloc[:3200, :]
test_age = age_feat.iloc[3200:, :]

train_gender = gender_feat.iloc[:3200, :]
test_gender = gender_feat.iloc[3200:, :]

train_area = area_feat.iloc[:3200, :]
test_area = area_feat.iloc[3200:, :]

# ---train---
age_le = LabelEncoder()
y_age = age_le.fit_transform(train_age.age2)
x_age = train_age.loc[:,'sum_fans':]
x_test = test_age.loc[:,'sum_fans':]


x_train, x_val, y_train, y_val = train_test_split(x_age, y_age, test_size=0.2, stratify=y_age, random_state=42)

#xgboost start here

dtest = xgb.DMatrix(x_test)
dval = xgb.DMatrix(x_val,label=y_val)
dtrain = xgb.DMatrix(x_train, label=y_train)

random_seed = 2016

params={
    'booster':'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
#     'scale_pos_weight': 189357.0/500588.0,
    'eval_metric': 'merror',
    'gamma':5.0,
    'max_depth':5,
#     'alpha':0.1,
#     'lambda':15,
    'subsample':0.7,
    'colsample_bytree':0.8,
    'min_child_weight':5,
    'eta': 0.005,
    'seed':random_seed,
    }

watchlist  = [(dval,'val'), (dtrain,'train')]
#watchlist  = [ (dtrain,'train')]
num_round = 2000
early_stop_rounds = 2000

model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=early_stop_rounds, verbose_eval=True)

import operator
%matplotlib inline
from matplotlib import pylab as plt

importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

#plt.figure()
#df.plot()
#df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
#plt.title('XGBoost Feature Importance')
#plt.xlabel('relative importance')

flist = df.feature.values
# predict age
age_le = LabelEncoder()
y_age = age_le.fit_transform(train_age.age2)
x_age = train_age[flist]
x_test = test_age[flist]

x_train, x_val, y_train, y_val = train_test_split(x_age, y_age, test_size=0.2, stratify=y_age, random_state=42)

TR=3200
#xgboost start here

dtest = xgb.DMatrix(x_test)
dval = xgb.DMatrix(x_val,label=y_val)
dtrain = xgb.DMatrix(x_train, label=y_train)

random_seed = 2016

params={
    'booster':'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
#     'scale_pos_weight': 189357.0/500588.0,
    'eval_metric': 'merror',
    'gamma':5.0,
    'max_depth':8,
#     'alpha':0.1,
     'lambda':50,
    'subsample':0.7,
    'colsample_bytree':0.9,
    'min_child_weight':5,
    'eta': 0.005,
    'seed':random_seed,
    }

watchlist  = [(dval,'val'), (dtrain,'train')]
#watchlist  = [ (dtrain,'train')]
num_round = 2000
early_stop_rounds = 1500

model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=early_stop_rounds, verbose_eval=True)

# ---predict---
age_preds = model.predict(dtest, ntree_limit=1500)
sub = pd.DataFrame()
sub['uid'] = merge_data.iloc[TR:]['uid']
sub['age'] = age_le.inverse_transform(age_preds.astype(int))
sub.to_csv('./data/age_sub.csv',index=False)
