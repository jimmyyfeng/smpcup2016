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
from collections import defaultdict

# ---read data---
path = './data/'

raw_feat = pd.read_csv(path + 'newfeat/merge_data.csv', sep=',', header = 'infer' )


# ---area feature---
area = ['∂´±±', 'ª™±±', 'ª™÷–', 'ª™∂´', 'Œ˜±±', 'Œ˜ƒœ', 'ª™ƒœ', 'æ≥Õ‚']

loc  = defaultdict(list)
loc[0] = ['∂´±±','¡…ƒ˛', 'º™¡÷', '∫⁄¡˙Ω≠',
          '…Ú—Ù', '¥Û¡¨','∞∞…Ω', '∏ßÀ≥','±æœ™', 'µ§∂´', 'Ωı÷›', '”™ø⁄', '∏∑–¬', '¡…—Ù', '≈ÃΩı', 'Ã˙¡Î', '≥Ø—Ù', '∫˘¬´µ∫',
         '≥§¥∫', 'º™¡÷', 'Àƒ∆Ω', '¡…‘¥', 'Õ®ªØ', '∞◊…Ω', 'À…‘≠', '∞◊≥«', '—”±ﬂ',
         'π˛∂˚±ı', '∆Î∆Îπ˛∂˚', 'º¶Œ˜', '∫◊∏⁄', 'À´—º…Ω', '¥Û«Ï', '“¡¥∫', 'º—ƒæÀπ', '∆ﬂÃ®∫”', 'ƒµµ§Ω≠', '∫⁄∫”', 'ÀÁªØ', '¥Û–À∞≤¡Î']

loc[1] = [ 'ª™±±','±±æ©', 'ÃÏΩÚ', '∫”±±', '…ΩŒ˜', 'ƒ⁄√…π≈',
         ' Øº“◊Ø','Ã∆…Ω', '«ÿª µ∫', '∫™µ¶', '–œÃ®', '±£∂®', '’≈º“ø⁄', '≥–µ¬', '≤◊÷›', '¿»∑ª', '∫‚ÀÆ',
         'Ã´‘≠', '¥ÛÕ¨', '—Ù»™', '≥§÷Œ', 'Ω˙≥«', 'À∑÷›', 'Ω˙÷–', '‘À≥«–√÷›', '¡Ÿ∑⁄', '¬¿¡∫',
          '∫Ù∫Õ∫∆Ãÿ' ,'∞¸Õ∑', 'Œ⁄∫£', '≥‡∑Â', 'Õ®¡…', '∂ı∂˚∂‡Àπ', '∫Ù¬◊±¥∂˚', '∞Õ—Âƒ◊∂˚', 'Œ⁄¿º≤Ï≤º', '–À∞≤', 'Œ˝¡÷π˘¿’', '∞¢¿≠…∆']

loc[2] = [ 'ª™÷–', '∫”ƒœ', '∫˛±±', '∫˛ƒœ',
         '÷£÷›', 'ø™∑‚', '¬Â—Ù', '∆Ω∂•…Ω', 'Ωπ◊˜', '∫◊±⁄', '–¬œÁ', '∞≤—Ù', 'Âß—Ù', '–Ì≤˝', '‰∫”', '»˝√≈œø','ƒœ—Ù', '…Ã«', '–≈—Ù', '÷‹ø⁄' '◊§¬ÌµÍ',
         'Œ‰∫∫', 'ª∆ Ø', 'œÂ∑Æ', ' Æ—ﬂ', 'æ£÷›', '“À≤˝', 'æ£√≈', '∂ı÷›', '–¢∏–', 'ª∆∏‘', 'œÃƒ˛', 'ÀÊ÷›', '∂˜ ©',
         '≥§…≥', '÷Í÷ﬁ', 'œÊÃ∂', '∫‚—Ù', '…€—Ù', '‘¿—Ù', '≥£µ¬', '’≈º“ΩÁ', '“Ê—Ù', '≥ª÷›', '”¿÷›', 'ª≥ªØ', '¬¶µ◊', 'œÊŒ˜']

loc[3] = [ 'ª™∂´','…œ∫£', 'Ω≠À’', '’„Ω≠', '∞≤ª’', '∏£Ω®', 'Ω≠Œ˜', '…Ω∂´','Ã®ÕÂ',
         'ƒœæ©', 'ŒﬁŒ˝', '–Ï÷›', '≥£÷›', 'À’÷›', 'ƒœÕ®', '¡¨‘∆∏€', 'ª¥∞≤', '—Œ≥«', '—Ô÷›', '’ÚΩ≠', 'Ã©÷›', 'Àﬁ«®',
        '∫º÷›', 'ƒ˛≤®', 'Œ¬÷›', 'ºŒ–À', '∫˛÷›', '…‹–À', 'Ωª™', '·È÷›', '÷€…Ω', 'Ã®÷›', '¿ˆÀÆ',
         '∫œ∑ ', 'Œﬂ∫˛', '∞ˆ≤∫', 'ª¥ƒœ', '¬Ì∞∞…Ω', 'ª¥±±', 'Õ≠¡Í', '∞≤«Ï', 'ª∆…Ω', '≥¸÷›', '∏∑—Ù', 'Àﬁ÷›', '≥≤∫˛', '¡˘∞≤', 'ŸÒ÷›', '≥ÿ÷›', '–˚≥«',
         '∏£÷›', 'œ√√≈', '∆ŒÃÔ', '»˝√˜', '»™÷›', '’ƒ÷›', 'ƒœ∆Ω', '¡˙—“', 'ƒ˛µ¬',
         'ƒœ≤˝', 'æ∞µ¬’Ú', '∆ºœÁ', 'æ≈Ω≠', '–¬”‡', '”•Ã∂', '∏”÷›', 'º™∞≤', '“À¥∫', '∏ß÷›', '…œ»ƒ',
         'º√ƒœ', '«‡µ∫', '◊Õ≤©', '‘Ê◊Ø', '∂´”™', '—ÃÃ®', 'Œ´∑ª', 'Õ˛∫£', 'º√ƒ˛', 'Ã©∞≤', '»’’’', '¿≥Œﬂ', '¡Ÿ“ ', 'µ¬÷›', '¡ƒ≥«', '±ı÷›', '∫ ‘Û']

loc[4] = [ 'Œ˜±±','…¬Œ˜', '∏ À‡', '«‡∫£', 'ƒ˛œƒ', '–¬ΩÆ',
         'Œ˜∞≤', 'Õ≠¥®', '±¶º¶', 'œÃ—Ù', 'Œºƒœ', '—”∞≤', '∫∫÷–', '”‹¡÷', '∞≤øµ', '…Ã¬Â',
         '¿º÷›', 'ºŒ”¯πÿ', 'Ω≤˝', '∞◊“¯', 'ÃÏÀÆ', 'Œ‰Õ˛','’≈“¥', '∆Ω¡π', 'æ∆»™', '«Ï—Ù', '∂®Œ˜', '¬§ƒœ', '¡Ÿœƒ', '∏ ƒœ',
         'Œ˜ƒ˛', '∫£∂´', '∫£±±', 'ª∆ƒœ', '∫£ƒœ', 'π˚¬Â', '”Ò ˜', '∫£Œ˜',
         '“¯¥®', ' Ø◊Ï…Ω', 'Œ‚÷“', 'πÃ‘≠', '÷–Œ¿',
         'Œ⁄¬≥ƒæ∆Î', 'øÀ¿≠¬Í“¿', 'Õ¬¬≥∑¨', 'π˛√‹', '∫ÕÃÔ', '∞¢øÀÀ’', 'ø¶ ≤', 'øÀ◊Œ¿’À’ø¬∂˚øÀ◊Œ', '∞Õ“Ùπ˘¿„√…π≈', '≤˝º™', '≤©∂˚À˛¿≠√…π≈', '“¡¿Áπ˛»¯øÀÀ˛≥«','∞¢¿’Ã©',
         ]

loc[5] = ['Œ˜ƒœ', '÷ÿ«Ï', 'Àƒ¥®', 'πÛ÷›', '‘∆ƒœ', 'Œ˜≤ÿ',
          '≥…∂º', '◊‘π±', '≈ ÷¶ª®', '„Ú÷›', 'µ¬—Ù', '√‡—Ù', 'π„‘™', 'ÀÏƒ˛', 'ƒ⁄Ω≠', '¿÷…Ω', 'ƒœ≥‰', '“À±ˆ', 'π„∞≤', '¥Ô÷›', '√º…Ω', '—≈∞≤', '∞Õ÷–', '◊ —Ù', '∞¢∞”', '∏ ◊Œ¡π…Ω',
         'πÛ—Ù', '¡˘≈ÃÀÆ', '◊Ò“Â', '∞≤À≥', 'Õ≠» ', '±œΩ⁄', '«≠Œ˜ƒœ', '«≠∂´ƒœ', '«≠ƒœ',
         '¿•√˜', '«˙æ∏', '”Òœ™', '±£…Ω', '’—Õ®', '¿ˆΩ≠', '∆’∂˝', '¡Ÿ≤◊', 'Œƒ…Ω', '∫Ï∫”', 'Œ˜À´∞Êƒ…', '≥˛–€', '¥Û¿Ì', 'µ¬∫Í', '≈≠Ω≠', 'µœ«Ï',
         '¿≠»¯', '≤˝∂º', '…Ωƒœ', '»’ø¶‘Ú', 'ƒ««˙', '∞¢¿Ô', '¡÷÷•']

loc[6] =['ª™ƒœ','π„∂´', '∫£ƒœ', 'π„Œ˜','œ„∏€','∞ƒ√≈', 
        'π„÷›', '…Ó€⁄', '÷È∫£', '…«Õ∑', '…ÿπÿ', '∑…Ω', 'Ω≠√≈', '’øΩ≠', '√Ø√˚', '’ÿ«Ï', 'ª›÷›', '√∑÷›', '…«Œ≤', '∫”‘¥', '—ÙΩ≠', '«Â‘∂', '∂´›∏', '÷–…Ω', '≥±÷›', 'Ω“—Ù', '‘∆∏°',
        'ƒœƒ˛', '¡¯÷›', 'π¡÷', 'Œ‡÷›', '±±∫£', '∑¿≥«∏€', '«’÷›', 'πÛ∏€', '”Ò¡÷', '∞Ÿ…´', '∫ÿ÷›', '∫”≥ÿ', '¿¥±ˆ', '≥Á◊Û',
        '∫£ø⁄', '»˝—«']

loc[7] = ['æ≥Õ‚','”¢π˙','∞Æ∂˚¿º','∫…¿º','±»¿˚ ±','¬¨…≠±§','∑®π˙','ƒ¶ƒ…∏Á',
         '∑“¿º','»µ‰','≈≤Õ˛','±˘µ∫','µ§¬Û','∑®¬ﬁ»∫µ∫',
         '“‚¥Û¿˚','ËÛµŸ∏‘',' •¬Ì¡¶≈µ','¬Ì∂˙À˚','Œ˜∞‡—¿','∆œÃ——¿','∞≤µ¿∂˚',
         '√¿π˙','º”ƒ√¥Û','ƒ´Œ˜∏Á','∞ÕŒ˜','∞¢∏˘Õ¢','÷«¿˚','ŒØƒ⁄»¿≠','∞Õƒ√¬Ì','π≈∞Õ']

featAreaPd = pd.DataFrame(columns=range(len(area)))
featAreaPd.columns = 'area_' + featAreaPd.columns.astype(str)
print featAreaPd
alen = len(area)


for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(len(area)):
        for l in loc[i]:
            tmpL[i] = tmpL[i] + tmpDict[l]
    featAreaPd = featAreaPd.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(featAreaPd.columns)), ignore_index=True)
	
	
# ---statistic feature---
topfsource = pd.read_csv(path + 'frequent/topfsource.csv', sep=',', header = 'infer' )
alen = topfsource.shape[0]
feat_topfsource = pd.DataFrame(columns=range(alen))
feat_topfsource.columns = 'topfsource_' + feat_topfsource.columns.astype(str)
for content in raw_feat['source_content']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topfsource['source'][i]]
    feat_topfsource = feat_topfsource.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topfsource.columns)), ignore_index=True)

	
topfwords = pd.read_csv(path + 'frequent/topfwords.csv', sep=',', header = 'infer' )
alen = topfwords.shape[0]
feat_topfwords = pd.DataFrame(columns=range(alen))
feat_topfwords.columns = 'topfwords_' + feat_topfwords.columns.astype(str)
for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topfwords['word'][i]]
    feat_topfwords = feat_topfwords.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topfwords.columns)), ignore_index=True)
   
   
topmidsource = pd.read_csv(path + 'frequent/topmidsource.csv', sep=',', header = 'infer' )
alen = topmidsource.shape[0]
feat_topmidsource = pd.DataFrame(columns=range(alen))
feat_topmidsource.columns = 'topmidsource_' + feat_topmidsource.columns.astype(str)
for content in raw_feat['source_content']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topmidsource['source'][i]]
    feat_topmidsource = feat_topmidsource.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topmidsource.columns)), ignore_index=True)
        
topmidwordcount = pd.read_csv(path + 'frequent/topmidwordcount.csv', sep=',', header = 'infer' )
alen = topmidwordcount.shape[0]
feat_topmidwordcount = pd.DataFrame(columns=range(alen))
feat_topmidwordcount.columns = 'topmidwordcount_' + feat_topmidwordcount.columns.astype(str)
for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topmidwordcount['word'][i]]
    feat_topmidwordcount = feat_topmidwordcount.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topmidwordcount.columns)), ignore_index=True)
   
   
topmsource = pd.read_csv(path + 'frequent/topmsource.csv', sep=',', header = 'infer' )
alen = topmsource.shape[0]
feat_topmsource = pd.DataFrame(columns=range(alen))
feat_topmsource.columns = 'topmsource_' + feat_topmsource.columns.astype(str)
for content in raw_feat['source_content']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topmsource['source'][i]]
    feat_topmsource = feat_topmsource.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topmsource.columns)), ignore_index=True) 
  

topoldsource = pd.read_csv(path + 'frequent/topoldsource.csv', sep=',', header = 'infer' )
alen = topoldsource.shape[0]
feat_topoldsource = pd.DataFrame(columns=range(alen))
feat_topoldsource.columns = 'topoldsource_' + feat_topoldsource.columns.astype(str)
for content in raw_feat['source_content']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topoldsource['source'][i]]
    feat_topoldsource = feat_topoldsource.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topoldsource.columns)), ignore_index=True)
  
  
  
topyoungsource = pd.read_csv(path + 'frequent/topyoungsource.csv', sep=',', header = 'infer' )
alen = topyoungsource.shape[0]
feat_topyoungsource = pd.DataFrame(columns=range(alen))
feat_topyoungsource.columns = 'topyoungsource_' + feat_topyoungsource.columns.astype(str)
for content in raw_feat['source_content']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topyoungsource['source'][i]]
    feat_topyoungsource = feat_topyoungsource.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topyoungsource.columns)), ignore_index=True)  
	
	
	
topoldwordcount = pd.read_csv(path + 'frequent/topoldwordcount.csv', sep=',', header = 'infer' )
alen = topoldwordcount.shape[0]
feat_topoldwordcount = pd.DataFrame(columns=range(alen))
feat_topoldwordcount.columns = 'topoldwordcount_' + feat_topoldwordcount.columns.astype(str)
for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topoldwordcount['word'][i]]
    feat_topoldwordcount = feat_topoldwordcount.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topoldwordcount.columns)), ignore_index=True)
 
 
topyoungwordcount = pd.read_csv(path + 'frequent/topyoungwordcount.csv', sep=',', header = 'infer' )
alen = topyoungwordcount.shape[0]
feat_topyoungwordcount = pd.DataFrame(columns=range(alen))
feat_topyoungwordcount.columns = 'topyoungwordcount_' + feat_topyoungwordcount.columns.astype(str)
for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topyoungwordcount['word'][i]]
    feat_topyoungwordcount = feat_topyoungwordcount.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topyoungwordcount.columns)), ignore_index=True)

topmwords = pd.read_csv(path + 'frequent/topmwords.csv', sep=',', header = 'infer' )
alen = topmwords.shape[0]
feat_topmwords = pd.DataFrame(columns=range(alen))
feat_topmwords.columns = 'topmwords_' + feat_topmwords.columns.astype(str)
for content in raw_feat['contents']:
    tmpDict  = defaultdict(int)
    for word in content.split():
        tmpDict[word] = tmpDict[word] + 1
    tmpL  = [0] * alen
    for i in range(alen):
        tmpL[i] = tmpDict[topmwords['word'][i]]
    feat_topmwords = feat_topmwords.append(pd.DataFrame(np.array(tmpL).reshape(1,alen), columns = list(feat_topmwords.columns)), ignore_index=True)


path_area = path + 'frequent/toplocwords/'
area_word = ['topdbword', 'tophbword', 'tophdword', 'tophnword', 'tophzword', 'topjwword', 'topxbword', 'topxnword']
feat_area_word = pd.DataFrame()
k = 0
for w in area_word:
    topword = pd.read_csv(path_area + w + '.csv', sep=',', header = 'infer' )
    len_topword = topword.shape[0]
    feat_topword = pd.DataFrame(columns=range(len_topword))
    feat_topword.columns = w + '_' + feat_topword.columns.astype(str)
    
    for content in raw_feat['contents']:
        tmpDict  = defaultdict(int)
        for word in content.split():
            tmpDict[word] = tmpDict[word] + 1
        tmpL  = [0] * len_topword
        for i in range(len_topword):
            tmpL[i] = tmpDict[topword['word'][i]]
        feat_topword = feat_topword.append(pd.DataFrame(np.array(tmpL).reshape(1,len_topword), columns = list(feat_topword.columns)), ignore_index=True)
    
    feat_topword[w + '_sum'] = feat_topword.sum(axis=1)
    feat_area_word = pd.concat([feat_area_word,feat_topword ],  axis = 1)
    print feat_topword.shape	
	
feat_topfsource['topfsource_sum'] = feat_topfsource.sum(axis=1)
feat_topmsource['topmsource_sum'] = feat_topmsource.sum(axis=1)
feat_topfwords['topfwords_sum'] = feat_topfwords.sum(axis=1)
feat_topmwords['topmwords_sum'] = feat_topmwords.sum(axis=1)

feat_topoldsource['topoldsource_sum'] = feat_topoldsource.sum(axis=1)
feat_topmidsource['topmidsource_sum'] = feat_topmidsource.sum(axis=1)
feat_topyoungsource['topyoungsource_sum'] = feat_topyoungsource.sum(axis=1)
feat_topoldwordcount['topoldwordcount_sum'] = feat_topoldwordcount.sum(axis=1)
feat_topmidwordcount['topmidwordcount_sum'] = feat_topmidwordcount.sum(axis=1)
feat_topyoungwordcount['topyoungwordcount_sum'] = feat_topyoungwordcount.sum(axis=1)


feat_sex = pd.concat([feat_topfsource,feat_topmsource,feat_topfwords, feat_topmwords ],  axis = 1)
feat_age = pd.concat([feat_topoldsource,feat_topmidsource,feat_topyoungsource, feat_topoldwordcount,feat_topmidwordcount,feat_topyoungwordcount ],  axis = 1)
feat_area = featAreaPd

# ---save statis feature--- 
feat_sex.to_csv(path+'newfeat/feat_sex.csv', header='infer', sep = ',', index = 0)
feat_age.to_csv(path+'newfeat/feat_age.csv', header='infer', sep = ',', index = 0)
feat_area.to_csv(path+'newfeat/feat_area.csv', header='infer', sep = ',', index = 0)
feat_area_word.to_csv(path+'newfeat/feat_area_word.csv', header='infer', sep = ',', index = 0)


# ---time feature---
def timePeriod(s):
    if ((s >'00:00') & (s<='03:00')):
        return 0
    elif((s >'03:00') & (s<='06:00')):
        return 1
    elif((s >'06:00') & (s<='09:00')):
        return 2
    elif((s >'09:00') & (s<='12:00')):
        return 3
    elif((s >'12:00') & (s<='15:00')):
        return 4
    elif((s >'15:00') & (s<='18:00')):
        return 5
    elif((s >'18:00') & (s<='21:00')):
        return 6
    elif((s >'21:00') & (s<='24:00')):
        return 7

train_path = path + 'train/' 
test_path = path + 'valid/'


train_status_lines = open(train_path + 'train_status.txt').readlines()
train_dict = dict()
early_flag = 0
for train_status in train_status_lines:
    r = train_status.split(',')
    s = r[4].split()
    
    if(len(s)==1):
        early_flag = 1    
    else:    
        t = timePeriod(s[1])
        if((r[0] in train_dict) == False):
            train_dict[r[0]] = defaultdict(int)
        train_dict[r[0]][t] = train_dict[r[0]][t] + 1
        if(early_flag==1):
            train_dict[r[0]][t] = train_dict[r[0]][t] + 1
            early_flag = 0
train_dict_pd = pd.DataFrame(train_dict).T.reset_index().rename(columns={"index": "uid"})
train_dict_pd = train_dict_pd.drop(None, axis=1)
train_dict_pd.columns = 'timePeriod_3hour_' + train_dict_pd.columns.astype(str)
train_dict_pd = train_dict_pd.rename(columns={"timePeriod_3hour_uid": "uid"})
train_dict_pd = train_dict_pd.fillna(0)

test_status_lines = open(test_path + 'valid_status.txt').readlines()
test_dict = dict()
early_flag = 0
for test_status in test_status_lines:
    r = test_status.split(',')
    s = r[4].split()
    
    if(len(s)==1):
        early_flag = 1    
    else:    
        t = timePeriod(s[1])
        if((r[0] in test_dict) == False):
            test_dict[r[0]] = defaultdict(int)
        test_dict[r[0]][t] = test_dict[r[0]][t] + 1
        if(early_flag==1):
            test_dict[r[0]][t] = test_dict[r[0]][t] + 1
            early_flag = 0
test_dict_pd = pd.DataFrame(test_dict).T.reset_index().rename(columns={"index": "uid"})
test_dict_pd = test_dict_pd.drop(None, axis=1)
test_dict_pd.columns = 'timePeriod_3hour_' + test_dict_pd.columns.astype(str)
test_dict_pd = test_dict_pd.rename(columns={"timePeriod_3hour_uid": "uid"})
test_dict_pd = test_dict_pd.fillna(0)

feat_time_3hour = pd.concat([train_dict_pd, test_dict_pd], axis = 0, ignore_index = True )

def timePerHour(s):
    if ((s >'00:00') & (s<='01:00')):
        return 0
    elif((s >'01:00') & (s<='02:00')):
        return 1
    elif((s >'02:00') & (s<='03:00')):
        return 2
    elif((s >'03:00') & (s<='04:00')):
        return 3
    elif((s >'04:00') & (s<='05:00')):
        return 4
    elif((s >'05:00') & (s<='06:00')):
        return 5
    elif((s >'06:00') & (s<='07:00')):
        return 6
    elif((s >'07:00') & (s<='08:00')):
        return 7
    elif((s >'08:00') & (s<='09:00')):
        return 8
    elif((s >'09:00') & (s<='10:00')):
        return 9
    elif((s >'10:00') & (s<='11:00')):
        return 10
    elif((s >'11:00') & (s<='12:00')):
        return 11
    elif((s >'12:00') & (s<='13:00')):
        return 12
    elif((s >'13:00') & (s<='14:00')):
        return 13
    elif((s >'14:00') & (s<='15:00')):
        return 14
    elif((s >'15:00') & (s<='16:00')):
        return 15
    elif((s >'16:00') & (s<='17:00')):
        return 16
    elif((s >'17:00') & (s<='18:00')):
        return 17
    elif((s >'18:00') & (s<='19:00')):
        return 18
    elif((s >'19:00') & (s<='20:00')):
        return 19
    elif((s >'20:00') & (s<='21:00')):
        return 20
    elif((s >'21:00') & (s<='22:00')):
        return 21
    elif((s >'22:00') & (s<='23:00')):
        return 22
    elif((s >'23:00') & (s<='24:00')):
        return 23

train_path = path + 'train/'
test_path = path + 'valid/'


train_status_lines = open(train_path + 'train_status.txt').readlines()
train_dict = dict()
early_flag = 0
for train_status in train_status_lines:
    r = train_status.split(',')
    s = r[4].split()
    
    if(len(s)==1):
        early_flag = 1    
    else:    
        t = timePerHour(s[1])
        if((r[0] in train_dict) == False):
            train_dict[r[0]] = defaultdict(int)
        train_dict[r[0]][t] = train_dict[r[0]][t] + 1
        if(early_flag==1):
            train_dict[r[0]][t] = train_dict[r[0]][t] + 1
            early_flag = 0
train_dict_pd = pd.DataFrame(train_dict).T.reset_index().rename(columns={"index": "uid"})
train_dict_pd = train_dict_pd.drop(None, axis=1)
train_dict_pd.columns = 'timePeriod_1hour_' + train_dict_pd.columns.astype(str)
train_dict_pd = train_dict_pd.rename(columns={"timePeriod_1hour_uid": "uid"})
train_dict_pd = train_dict_pd.fillna(0)

test_status_lines = open(test_path + 'valid_status.txt').readlines()
test_dict = dict()
early_flag = 0
for test_status in test_status_lines:
    r = test_status.split(',')
    s = r[4].split()
    
    if(len(s)==1):
        early_flag = 1    
    else:    
        t = timePerHour(s[1])
        if((r[0] in test_dict) == False):
            test_dict[r[0]] = defaultdict(int)
        test_dict[r[0]][t] = test_dict[r[0]][t] + 1
        if(early_flag==1):
            test_dict[r[0]][t] = test_dict[r[0]][t] + 1
            early_flag = 0
test_dict_pd = pd.DataFrame(test_dict).T.reset_index().rename(columns={"index": "uid"})
test_dict_pd = test_dict_pd.drop(None, axis=1)
test_dict_pd.columns = 'timePeriod_1hour_' + test_dict_pd.columns.astype(str)
test_dict_pd = test_dict_pd.rename(columns={"timePeriod_1hour_uid": "uid"})
test_dict_pd = test_dict_pd.fillna(0)

feat_time_1hour = pd.concat([train_dict_pd, test_dict_pd], axis = 0, ignore_index = True )


# ---save time feature---
feat_time_3hour.to_csv(path+'newfeat/feat_time_3hour.csv', header='infer', sep = ',', index = 0)
feat_time_1hour.to_csv(path+'newfeat/feat_time_1hour.csv', header='infer', sep = ',', index = 0)
