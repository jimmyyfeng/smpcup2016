#coding=utf-8
#author:不思蜀
#：20160917
import jieba
from gensim.models import word2vec
import logging
import sys
import time
reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == '__main__':

    fr = open('./data/valid/valid_nolabel.txt', 'r')
    user2 = []
    line = fr.readline()
    while line:
        user2.append(line.split("\r\n")[0])
        line = fr.readline()
    fr.close()
    print len(user2), "----------用户个数---------------"

    fru = open('./data/train/lb_and_count_weibo.txt', 'r')
    uuu = dict()
    line = fru.readline()
    p = 0
    while line:
        uuu[line.split("||")[0]] = line.split("||")[4]
        line = fru.readline()
        p += 1
        if p % 10000 == 0:
            print p
    fru.close()
    print uuu
    print len(uuu.keys())
    fru = open('./data/valid/valid_status.txt', 'r')
    line = fru.readline()
    while line:
        uc = line.split(",")[0]
        if uc  in user2 :
            if uc in uuu.keys():
                uuu[uc] += 1
            else:uuu[uc] = 1
        line = fru.readline()
    print uuu
    print len(uuu.keys())
    fru.close()


    print "load w2c model"
    model = word2vec.Word2Vec.load(u"20160913_w2c_train.model")
    print "load OK"
    fr = open('./data/train/train_labels.txt','r')
    user = []
    line = fr.readline()
    while line:
        user.append(line.split("||")[0])
        line = fr.readline()
    fr.close()
    print len(user),"----------用户个数---------------"
    fr = open('./data/train/train_status.txt','r')
    line = fr.readline()
    m = []
    for i in range(4440):
        m.append([])
        for j in range(128):
            m[i].append(0)
    i = 0
    while line:
            for c in range(len(user)):
                if user[c] == line.split(",")[0]:
                    break
            i += 1
            sen = line.split(",")[5]
            #print sen
            word_list = '\t'.join(jieba.cut(sen.replace(" ",""), cut_all=False)).split("\t")
            #print word_list
            for w in word_list:
                if w in model:
                    m[c] += model[w]/len(word_list)
                    #print model[w]
            line = fr.readline()
            if i % 1000 == 0:
                print i,"行已计算-----train"
            # if i == 6000:
            #     break
    print "---------------TRAIN SET-----------------------"


    fr = open('./data/valid/valid_status.txt', 'r')
    line = fr.readline()
    i = 0
    while line:
        for c in range(len(user2)):
            if user2[c] == line.split(",")[0]:
                break
        i += 1
        sen = line.split(",")[5]
        # print sen
        word_list = '\t'.join(jieba.cut(sen.replace(" ", ""), cut_all=False)).split("\t")
        # print word_list
        for w in word_list:
            if w in model:
                m[c+3200] += model[w] / len(word_list)
                # print model[w]
        line = fr.readline()
        if i % 1000 == 0:
            print i, "行已计算-----test"
        # if i == 3000:
        #     break
    print "-------------TEST SET-------------------------"

    fr.close()
    fr = open('./data/w2v_sentence.txt','w')
    for k in range(3200):
        s = str(user[k])
        for wq in m[k]:
            s = s + "," + str(wq/float(uuu[user[k]]))
        fr.write(s+"\n")
    for k in range(1240):
        s = str(user2[k])
        for wq in m[k+3200]:
            s = s + "," + str(wq/float(uuu[user2[k]]))
        fr.write(s+"\n")
    print "--------------------------------------"
    print user
    print i,user[1]
    print "------------------写完了---------------------"

