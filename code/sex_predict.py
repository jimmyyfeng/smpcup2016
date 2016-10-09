import numpy as np
import pandas as pd
import re
import xgboost as xgb
from sklearn import svm
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold


#读入训练测试数据，合并
train_data = pd.read_csv('./data/train/train_labels.txt',sep=u'|',header=None).dropna(1)
train_data.columns = ['uid','sex','age','location']
test_data = pd.read_csv('./data/valid/valid_nolabel.txt',sep=u'|',header=None).dropna(1)
test_data.columns = ['uid']
total_data = pd.concat([train_data,test_data],axis=0)



#读入训练测试info数据，合并
train_data_info = pd.read_csv('./data/train/train_info.txt',sep=u'|',header=None).dropna(1)
train_data_info.columns = ['uid','name','image']
train_data_info = train_data_info.drop_duplicates()
test_data_info = pd.read_csv('./data/valid/valid_info.txt',sep=u'|',header=None).dropna(1)
test_data_info.columns = ['uid','name','image']
test_data_info = test_data_info.drop_duplicates()
total_data_info = pd.concat([train_data_info,test_data_info],axis=0)
total_data_info = total_data_info.drop_duplicates('uid')



#读入训练测试links数据，合并
links = []
for i, line in enumerate(open('./data/train/train_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'sum_fans':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
train_data_links = pd.DataFrame(links)
train_data_links = train_data_links.drop_duplicates()


links = []
for i, line in enumerate(open('./data/valid/valid_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'sum_fans':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
test_data_links = pd.DataFrame(links)
test_data_links = test_data_links.drop_duplicates()

total_data_links = pd.concat([train_data_links,test_data_links],axis=0)



#读入训练测试status数据，合并
status = []
for i, line in enumerate(open('./data/train/train_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
train_data_status = pd.DataFrame(status)


status = []
for i, line in enumerate(open('./data/valid/valid_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
test_data_status = pd.DataFrame(status)

total_data_status = pd.concat([train_data_status,test_data_status],axis=0)



#合并题目给的几个表数据
merge_data = pd.merge(total_data,total_data_info,on='uid',how='left')
merge_data = pd.merge(merge_data,total_data_links,on='uid',how='left')
merge_data.index = range(len(merge_data))
##################################################################################



total_data_status['retweet'] = total_data_status.sta.map(lambda s:int(s.split(',')[0]))
total_data_status['review'] = total_data_status.sta.map(lambda s:int(s.split(',')[1]))
total_data_status['source'] = total_data_status.sta.map(lambda s:s.split(',')[2])
total_data_status['time'] = total_data_status.sta.map(lambda s:s.split(',')[3])
total_data_status['content'] = total_data_status.sta.map(lambda s:','.join(s.split(',')[4:]))
contents = total_data_status.groupby('uid')['content'].agg(lambda lst:' '.join(lst))
merge_data['contents'] = merge_data.uid.map(contents)
merge_data['sum_content'] = merge_data.uid.map(total_data_status.groupby('uid').size())



#统计特征
merge_data['max_retweet'] = merge_data.uid.map(total_data_status.groupby('uid')['retweet'].agg('max'))
merge_data['max_review'] = merge_data.uid.map(total_data_status.groupby('uid')['review'].agg('max'))
merge_data['min_retweet'] = merge_data.uid.map(total_data_status.groupby('uid')['retweet'].agg('min'))
merge_data['min_review'] = merge_data.uid.map(total_data_status.groupby('uid')['review'].agg('min'))
merge_data['median_retweet'] = merge_data.uid.map(total_data_status.groupby('uid')['retweet'].agg('median'))
merge_data['median_review'] = merge_data.uid.map(total_data_status.groupby('uid')['review'].agg('median'))
merge_data['mean_retweet'] = merge_data.uid.map(total_data_status.groupby('uid')['retweet'].agg('mean'))
merge_data['mean_review'] = merge_data.uid.map(total_data_status.groupby('uid')['review'].agg('mean'))
merge_data['std_retweet'] = merge_data.uid.map(total_data_status.groupby('uid')['retweet'].agg('std'))
merge_data['std_review'] = merge_data.uid.map(total_data_status.groupby('uid')['review'].agg('std'))


#location地区映射词表
d = {'石家庄': '华北',
 '子陵庙': '华东',
 '深圳': '华南',
 '广州': '华南',
 '宝安': '华南',
 '刘庄': '华中',
 '沙市': '华中',
 '武汉': '华中',
 '襄阳': '华中',
 '安陆': '华中',
 '荆门': '华中',
 '西安': '西北',
 '银川': '西北',
 '成都': '西南',
 '绵阳': '西南',
 '上海': '华东',
 '云南': '西南',
 '内蒙古': '华北',
 '北京': '华北',
 '台湾': '华东',
 '吉林': '东北',
 '四川': '西南',
 '天津': '华北',
 '宁夏': '西北',
 '安徽': '华东',
 '山东': '华东',
 '山西': '华北',
 '辽宁': '东北',
 '重庆': '西南',
 '陕西': '西北',
 '青海': '西北',
 '香港': '华南',
 '黑龙江': '东北',
 '长白': '东北',
 '丹东': '东北',
 '大庸桥': '东北',
 '沈阳': '东北',
 '大连': '东北',
 '抚顺': '东北',
 '石家庄': '华北',
 '朝阳': '华北',
 '广东': '华南',
 '广西': '华南',
 '新疆': '西北',
 '江苏': '华东',
 '江西': '华东',
 '河北': '华北',
 '河南': '华中',
 '浙江': '华东',
 '海南': '华南',
 '湖北': '华中',
 '湖南': '华中',
 '澳门': '华南',
 '甘肃': '西北',
 '福建': '华东',
 '西藏': '西南',
 '贵州': '西南',
}


#将location和age转化成需要提交的范围
def trans_loc(s):
    if pd.isnull(s):
        return s
    s = s.split(' ')[0]
    if s == 'None':
        return '华北'
    if s == '海外':
        return s
    return d[s]

def trans_age(age):
    if pd.isnull(age):
        return age
    if age <=1979:
        return "-1979"
    elif age<=1989:
        return "1980-1989"
    else:
        return "1990+"



merge_data['location2'] = merge_data['location'].map(trans_loc)
merge_data['age2'] = merge_data['age'].map(trans_age)

src_lst = total_data_status.groupby('uid')['source'].agg(lambda lst:' '.join(lst))
merge_data['source_content'] = merge_data['uid'].map(src_lst) 

keys = '|'.join(d.keys())
merge_data['source_province'] = merge_data['source_content'].map(lambda s:' '.join(re.findall(keys,s)))
merge_data['num_province'] = merge_data['contents'].map(lambda s:' '.join(re.findall(keys,s)))

d = defaultdict(lambda :'空',d)
tokenizer = lambda line: [d[w] for w in line.split(' ')]
tfv = TfidfVectorizer(tokenizer=tokenizer,norm=False, use_idf=False, smooth_idf=False, sublinear_tf=False)
X_all_sp = tfv.fit_transform(merge_data['num_province'])
sum_province = X_all_sp.toarray()
for i in range(sum_province.shape[1]):
    merge_data['sum_province_%d'%i] = sum_province[:,i]



length = total_data_status.groupby('uid')['content'].agg(lambda lst:np.mean([len(s.split(' ')) for s in lst]))
merge_data['max_content_len'] = merge_data['uid'].map(length)
length = total_data_status.groupby('uid')['content'].agg(lambda lst:np.min([len(s.split(' ')) for s in lst]))
merge_data['min_content_len'] = merge_data['uid'].map(length)
length = total_data_status.groupby('uid')['content'].agg(lambda lst:np.max([len(s.split(' ')) for s in lst]))
merge_data['mean_content_len'] = merge_data['uid'].map(length)

merge_data['name_len'] = merge_data.name.map(lambda s:s if pd.isnull(s) else len(re.sub(r'[\u4e00-\u9fff]+','',s)))




def num_missing(x):    
    return sum(x.isnull())  

merge_data['num_missing'] = merge_data.apply(num_missing, axis=1) 

#rank特征
merge_data['rank_sum_content'] = merge_data['sum_content'].rank(method='max')
merge_data['rank_sum_fans'] = merge_data['sum_fans'].rank(method='max')
merge_data['rank_mean_retweet'] = merge_data['mean_retweet'].rank(method='max')
merge_data['rank_mean_review'] = merge_data['mean_review'].rank(method='max')
merge_data['rank_num_missing'] = merge_data['num_missing'].rank(method='max')


#导入使用tfidf特征训练的模型的预测结果（采用stacking融合，把预测结果作为新特征加进模型）
tfidf_stacking = pd.read_csv('./data/newfeat/stack_new.csv')
merge_data = pd.concat([merge_data,tfidf_stacking],axis=1)

#按小时划分，统计每个用户每3个小时发的微博数量
#feat_time_1hour = pd.read_csv('./data/newfeat/feat_time_1hour.csv')
#merge_data = pd.merge(merge_data,feat_time_1hour,on='uid',how='left')

feat_time_3hour = pd.read_csv('./data/newfeat/feat_time_3hour.csv')
merge_data = pd.merge(merge_data,feat_time_3hour,on='uid',how='left')

#导入使用word2vec特征训练的模型的预测结果
w2v_stacking = pd.read_csv('./data/newfeat/w2v_prob1.csv')
merge_data = pd.merge(merge_data,w2v_stacking,on='uid',how='left')

newmerge_feat1 = pd.read_csv('./data/newfeat/newmerge_feat.csv')
merge_data = pd.merge(merge_data,newmerge_feat1,on='uid',how='left')

feat_area1 = pd.read_csv('./data/newfeat/feat_area.csv')
merge_data = pd.merge(merge_data,feat_area1,on='uid',how='left')

#########################################################################################
cols = '|'.join(['twts_len','name_len','sum_province','sum_fans',
                'age_','sex_','loc_',
               'mean_retweet','sum_content','mean_review','num_missing',
                 'w2v_f_prob','w2v_m_prob','w2v_young_prob','w2v_old_prob','w2v_mid_prob',
                 'max_retweet','min_retweet','max_review','min_review',
                 'rank_sum_content','rank_sum_fans','rank_mean_retweet','rank_mean_review','rank_num_missing',
                 'timePeriod_3hour_0','timePeriod_3hour_1','timePeriod_3hour_2','timePeriod_3hour_3',
                 'timePeriod_3hour_4','timePeriod_3hour_5','timePeriod_3hour_6','timePeriod_3hour_7',
                 'name_isnull','image_isnull','fans_isnull','retweet_isnull','review_isnull',
                 'area_0','area_1','area_2','area_3','area_4','area_5','area_6','area_7'
                 ])
cols = [c for c in merge_data.columns if re.match(cols,c)]

age_le = LabelEncoder()
ys = {}
ys['age'] = age_le.fit_transform(merge_data.iloc[:3200]['age2'])

loc_le = LabelEncoder()
ys['loc'] = loc_le.fit_transform(merge_data.iloc[:3200]['location2'])

sex_le = LabelEncoder()
ys['sex'] = sex_le.fit_transform(merge_data.iloc[:3200]['sex'])


merge_data = merge_data.fillna(0)
task = ['sub']


TR = 3200
TE = 1240
X_all = merge_data[cols]
X = X_all[:TR]
prds = []





##############################
#年龄预测部分
label = 'age'
print('='*20)
print(label)
print('='*20)
y = ys[label]



n_trees = 500
params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "eval_metric": "merror",
    "num_class":3,
    'max_depth':8,
    'min_child_weight':2.5,
    'subsample':0.4,
    'colsample_bytree':1,
    'gamma':2.5,
    "eta": 0.01,
    "lambda":1,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for i,(tr,va) in enumerate(StratifiedKFold(y,n_folds=5)):
        print('stack:%d/%d'%(i+1,5))
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_va, y_va)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst1 = xgb.train(params, dtrain, 377, evals=watchlist,verbose_eval=20)
        
        
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst1 = xgb.train(params, dtrain, n_trees, evals=watchlist,
                     verbose_eval=100)
    prds.append(bst1.predict(dtest))


##########################
#性别预测部分
label = 'sex'
print('='*20)
print(label)
print('='*20)
y = ys[label]



n_trees = 429
params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "error",
    'max_depth':4,
    'min_child_weight':1.5,
    'subsample':1,
    'colsample_bytree':1,
    'gamma':4,
    "eta": 0.01,
    "lambda":3,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for i,(tr,va) in enumerate(StratifiedKFold(y,n_folds=5)):
        print('stack:%d/%d'%(i+1,5))
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_va, y_va)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst2 = xgb.train(params, dtrain, 327, evals=watchlist,verbose_eval=20)
        
        
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst2 = xgb.train(params, dtrain, n_trees, evals=watchlist,
                    verbose_eval=100)
    _prd = bst2.predict(dtest)
    prd = np.zeros((len(_prd),2))
    prd[:,1] = _prd
    prd[:,0] = 1 - prd[:,1]
    prds.append(prd)





########################
#地区预测部分
label = 'loc'
print('='*20)
print(label)
print('='*20)
y = ys[label]



n_trees = 616
params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "eval_metric": "merror",
    "num_class":8,
    'max_depth':5,
    'min_child_weight':2.5,
    'subsample':0.4,
    'colsample_bytree':1,
    'gamma':2.5,
    "eta": 0.01,
    "lambda":1,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for i,(tr,va) in enumerate(StratifiedKFold(y,n_folds=5)):
        print('stack:%d/%d'%(i+1,5))
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_va, y_va)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst3 = xgb.train(params, dtrain, 443, evals=watchlist,verbose_eval=20)
       
        
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst3 = xgb.train(params, dtrain, n_trees, evals=watchlist,
                     verbose_eval=100)
    prds.append(bst3.predict(dtest))


#########################
#生成提交结果
if 'sub' in task:
    sub = pd.DataFrame()
    sub['uid'] = merge_data.iloc[TR:]['uid']
    n = len(sub)
    sub['gender'] = sex_le.inverse_transform(prds[1].argmax(axis=1))
    sub.to_csv('./data/gender_sub.csv',index=False)
