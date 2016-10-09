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


path = './data/'

#∂¡»Î—µ¡∑≤‚ ‘ ˝æ›£¨∫œ≤¢
train_data = pd.read_csv( path + 'train/train_labels.txt',sep=u'|',header=None).dropna(1)
train_data.columns = ['uid','sex','age','location']
test_data = pd.read_csv(path + 'valid/valid_nolabel.txt',sep=u'|',header=None).dropna(1)
test_data.columns = ['uid']
total_data = pd.concat([train_data,test_data],axis=0)



#∂¡»Î—µ¡∑≤‚ ‘info ˝æ›£¨∫œ≤¢
train_data_info = pd.read_csv( path + 'train/train_info.txt',sep=u'|',header=None).dropna(1)
train_data_info.columns = ['uid','name','image']
train_data_info = train_data_info.drop_duplicates('uid')
test_data_info = pd.read_csv(path + 'valid/valid_info.txt',sep=u'|',header=None).dropna(1)
test_data_info.columns = ['uid','name','image']
test_data_info = test_data_info.drop_duplicates('uid')
total_data_info = pd.concat([train_data_info,test_data_info],axis=0)



#∂¡»Î—µ¡∑≤‚ ‘links ˝æ›£¨∫œ≤¢
links = []
for i, line in enumerate(open(path + 'train/train_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'sum_fans':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
train_data_links = pd.DataFrame(links)
train_data_links = train_data_links.drop_duplicates()

links = []
for i, line in enumerate(open(path + 'valid/valid_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'sum_fans':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
test_data_links = pd.DataFrame(links)
test_data_links = test_data_links.drop_duplicates()

total_data_links = pd.concat([train_data_links,test_data_links],axis=0)



#∂¡»Î—µ¡∑≤‚ ‘status ˝æ›£¨∫œ≤¢
status = []
for i, line in enumerate(open(path + 'train/train_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
train_data_status = pd.DataFrame(status)

status = []
for i, line in enumerate(open(path + 'valid/valid_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
test_data_status = pd.DataFrame(status)

total_data_status = pd.concat([train_data_status,test_data_status],axis=0)



#∫œ≤¢Ã‚ƒø∏¯µƒº∏∏ˆ±Ì ˝æ›
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



#Õ≥º∆Ãÿ’˜
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


#locationµÿ«¯”≥…‰¥ ±Ì
d = {' Øº“◊Ø': 'ª™±±',
 '◊”¡Í√Ì': 'ª™∂´',
 '…Ó€⁄': 'ª™ƒœ',
 'π„÷›': 'ª™ƒœ',
 '±¶∞≤': 'ª™ƒœ',
 '¡ı◊Ø': 'ª™÷–',
 '…≥ –': 'ª™÷–',
 'Œ‰∫∫': 'ª™÷–',
 'œÂ—Ù': 'ª™÷–',
 '∞≤¬Ω': 'ª™÷–',
 'æ£√≈': 'ª™÷–',
 'Œ˜∞≤': 'Œ˜±±',
 '“¯¥®': 'Œ˜±±',
 '≥…∂º': 'Œ˜ƒœ',
 '√‡—Ù': 'Œ˜ƒœ',
 '…œ∫£': 'ª™∂´',
 '‘∆ƒœ': 'Œ˜ƒœ',
 'ƒ⁄√…π≈': 'ª™±±',
 '±±æ©': 'ª™±±',
 'Ã®ÕÂ': 'ª™∂´',
 'º™¡÷': '∂´±±',
 'Àƒ¥®': 'Œ˜ƒœ',
 'ÃÏΩÚ': 'ª™±±',
 'ƒ˛œƒ': 'Œ˜±±',
 '∞≤ª’': 'ª™∂´',
 '…Ω∂´': 'ª™∂´',
 '…ΩŒ˜': 'ª™±±',
 '¡…ƒ˛': '∂´±±',
 '÷ÿ«Ï': 'Œ˜ƒœ',
 '…¬Œ˜': 'Œ˜±±',
 '«‡∫£': 'Œ˜±±',
 'œ„∏€': 'ª™ƒœ',
 '∫⁄¡˙Ω≠': '∂´±±',
 '≥§∞◊': '∂´±±',
 'µ§∂´': '∂´±±',
 '¥Û”π«≈': '∂´±±',
 '…Ú—Ù': '∂´±±',
 '¥Û¡¨': '∂´±±',
 '∏ßÀ≥': '∂´±±',
 ' Øº“◊Ø': 'ª™±±',
 '≥Ø—Ù': 'ª™±±',
 'π„∂´': 'ª™ƒœ',
 'π„Œ˜': 'ª™ƒœ',
 '–¬ΩÆ': 'Œ˜±±',
 'Ω≠À’': 'ª™∂´',
 'Ω≠Œ˜': 'ª™∂´',
 '∫”±±': 'ª™±±',
 '∫”ƒœ': 'ª™÷–',
 '’„Ω≠': 'ª™∂´',
 '∫£ƒœ': 'ª™ƒœ',
 '∫˛±±': 'ª™÷–',
 '∫˛ƒœ': 'ª™÷–',
 '∞ƒ√≈': 'ª™ƒœ',
 '∏ À‡': 'Œ˜±±',
 '∏£Ω®': 'ª™∂´',
 'Œ˜≤ÿ': 'Œ˜ƒœ',
 'πÛ÷›': 'Œ˜ƒœ',
}


#Ω´location∫Õage◊™ªØ≥…–Ë“™Ã·Ωªµƒ∑∂Œß
def trans_loc(s):
    if pd.isnull(s):
        return s
    s = s.split(' ')[0]
    if s == 'None':
        return 'ª™±±'
    if s == '∫£Õ‚':
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

d = defaultdict(lambda :'ø’',d)
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

#rankÃÿ’˜
merge_data['rank_sum_content'] = merge_data['sum_content'].rank(method='max')
merge_data['rank_sum_fans'] = merge_data['sum_fans'].rank(method='max')
merge_data['rank_mean_retweet'] = merge_data['mean_retweet'].rank(method='max')
merge_data['rank_mean_review'] = merge_data['mean_review'].rank(method='max')
merge_data['rank_num_missing'] = merge_data['num_missing'].rank(method='max')


merge_data.to_csv(path+'newfeat/merge_data.csv', header='infer', sep = ',', index = 0)
