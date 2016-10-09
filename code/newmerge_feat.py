import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
import operator
from matplotlib import pylab as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures

feat = pd.read_csv('./data/newfeat/merge_data.csv')
tfidf_prob = pd.read_csv('./data/newfeat/stack_new.csv')

newfeat = feat[['sex', 'age2', 'location2', 'uid', 'name', 'sum_fans', 'sum_content', 
                'min_content_len', 'mean_content_len', 'max_content_len',
               'mean_review', 'mean_retweet']]
newfeat['sum_review'] = newfeat.sum_content*newfeat.mean_review
newfeat['sum_retweet'] = newfeat.sum_content*newfeat.mean_retweet
newfeat['name_isnull'] = newfeat.name.isnull().astype(int)
newfeat['image_isnull'] = feat.image.isnull().astype(int)
newfeat['fans_isnull'] = feat.fans.isnull().astype(int)
newfeat['retweet_isnull'] = newfeat.sum_retweet.replace(0, np.nan).isnull().astype(int)
newfeat['review_isnull'] = newfeat.sum_review.replace(0, np.nan).isnull().astype(int)
newfeat = pd.concat([newfeat, tfidf_prob], axis=1)

newfeat.to_csv('./data/newfeat/newmerge_feat.csv')
