import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold

w2vfeat = pd.read_csv('./data/w2v_sentence1.txt', header=None)
newfeat = pd.read_csv('./data/newfeat/newmerge_feat.csv')

stacks_name = []
stacks_name += ['%s_%d'%('dim',i) for i in range(129)]
stacks_name[0] = 'uid'
w2vfeat.columns = stacks_name

w2vfeat['uid'] = [i for i in newfeat.uid]
train_label = newfeat[['uid', 'sex', 'age2', 'location2']].iloc[:3200, :]

# predict age_prob
train_age = w2vfeat.iloc[:3200, :]
test_age = w2vfeat.iloc[3200:, :]

age_le = LabelEncoder()
y_age = np.array(age_le.fit_transform(train_label.age2))
x_age = np.array(train_age.loc[:,'sum_fans':])
x_test = np.array(test_age.loc[:,'sum_fans':])

new_train = np.zeros((3200,3))
new_test = np.zeros((1240,3))
dtest = xgb.DMatrix(x_test)
y = y_age

random_seed = 2016
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

params={
    'booster':'gbtree',
    'objective': 'multi:softprob',
    'num_class': 3,
#     'scale_pos_weight': 0.8,
    'eval_metric': 'merror',
    'gamma':0.25,
    'max_depth':8,
#     'alpha':0.1,
    'lambda':100,
    'subsample':0.8,
    'colsample_bytree':0.9,
    'min_child_weight':4,
    'eta': 0.05,
    'seed':random_seed,
    }

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    dtrain = xgb.DMatrix(x_age[trainid],y[trainid])
    dval = xgb.DMatrix(x_age[valid],y[valid])
    bst = xgb.train(params, dtrain, num_boost_round=150)
    new_train[valid] = bst.predict(dval)
    new_test += bst.predict(dtest)


new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('age_prob',i) for i in range(3)]

stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

# predict gender_prob
train_gender = w2veat.iloc[:3200, :]
test_gender = w2vfeat.iloc[3200:, :]

gender_le = LabelEncoder()
y_gender = np.array(gender_le.fit_transform(train_label.sex))
x_gender = np.array(train_gender.loc[:,'sum_fans':])
x_test = np.array(test_gender.loc[:,'sum_fans':])

new_train = np.zeros((3200, 2))
new_test = np.zeros((1240, 2))
dtest = xgb.DMatrix(x_test)
y = y_gender

random_seed = 2016
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

params={
    'booster':'gbtree',
    'objective': 'multi:softprob',
    'num_class': 2,
    'scale_pos_weight': 0.8,
    'eval_metric': 'merror',
    'gamma':0.25,
    'max_depth':6,
#     'alpha':0.1,
    'lambda':100,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'min_child_weight':4,
    'eta': 0.05,
    'seed':random_seed,
    }

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    dtrain = xgb.DMatrix(x_age[trainid],y[trainid])
    dval = xgb.DMatrix(x_age[valid],y[valid])
    bst = xgb.train(params, dtrain, num_boost_round=200)
    new_train[valid] = bst.predict(dval)
    new_test += bst.predict(dtest)


new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('gender_prob',i) for i in range(2)]

stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

prob_feat = pd.concat([age_stacks, gender_stacks, area_stacks], axis=1)
prob_feat.columns = ['w2v_old_porb', 'w2v_mid_prob', 'w2v_young_prob', 'w2v_f_prob', 'w2v_m_prob']
prob_feat['uid'] = newfeat.uid
prob_feat.to_csv('./data/newfeat/w2v_prob1.csv', index=0)
