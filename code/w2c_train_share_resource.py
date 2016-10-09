#coding=utf-8
#author:不思蜀
#：20160901
#train_status.txt 中分享来源的聚类  网盘 分享 客户端 网站 手机
import jieba
from gensim.models import word2vec
import logging
import sys
import time
reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == '__main__':
    # fr = open('train/train_status.txt','r')
    # fr2 = open('train/jieba_share_status.txt','w')
    # line = fr.readline()
    # i = 0
    # while line:
    #     # fr2.write(' '.join(jieba.cut(line.split(",")[3],cut_all=False)))
    #     fr2.write(line.split(",")[5])
    #     line = fr.readline()
    # fr = open('test/test_status.txt','r')
    # line = fr.readline()
    # i = 0
    # while line:
    #     # fr2.write(' '.join(jieba.cut(line.split(",")[3],cut_all=False)))
    #     fr2.write(line.split(",")[5])
    #     line = fr.readline()
    # fr = open('train/unlabeled_statuses.txt','r')
    # line = fr.readline()
    # i = 0
    # while line:
    #         # fr2.write('  '.join(jieba.cut(line.split(",")[3],cut_all=False)))
    #         i += 1
    #         fr2.write(line.split(",")[5])
    #         if i % 500000 == 0:
    #             print i,line.split(",")[5]
    #         line = fr.readline()


    # fr = open('train/jieba_share_status.txt', 'r')
    # fr2 = open('train/w2c_20160913_term.txt', 'w')
    # line = fr.readline()
    # i = 0
    # print "开始分词",time.localtime()
    # while line:
    #     fr2.write('  '.join(jieba.cut(line, cut_all=False)))
    #     i += 1
    #     if i % 100000 == 0:
    #
    #         print i,'  '.join(jieba.cut(line, cut_all=False))
    #         print time.localtime()
    #
    #     line = fr.readline()
    # fr2.close()
    # print "分词结束---开始W2C训练",time.localtime()
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentences = word2vec.Text8Corpus("train/w2c_20160913_term.txt")  # 加载语料
    # print "W2C训练结束---准备保存",time.localtime()
    # print "---训练ing---已经加载好语料"
    # print sentences
    # model = word2vec.Word2Vec(sentences, size=128)
    # model.save(u"20160913_w2c_train.model")#22:36 时间
    # print "W2C保存完毕",time.localtime()

    print "load w2c model"
    model = word2vec.Word2Vec.load(u"20160913_w2c_train.model")
    print "load OK"
    
    print model[u'北京']

    #
    # # y4 = model.doesnt_match(u"安卓 三星 手机 iphone".split())
    # # print y4
    # fr = open('20160901_weibo_text_3000k_user.txt','r')
    # for i in range(50):
    #     print fr.readline()
