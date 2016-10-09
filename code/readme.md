data document：
	--- train : 原始训练集
	--- valid : 原始验证集
	--- frequent : 高频词文件
        ——- feature_v1 : baseline特征
	--- newfeat : 存放生成的数据特征文件
        ——- w2v_sentence(1) : 每个用户的平均向量（对所有博文求平均向量）
        ——- age_sub : age的预测结果
        ——— sex_sub : sex的预测结果  
        ——- location_sub : location的预测结果
        ——— final_sub : 最终提交结果



code document:

——- tfidf.py: 提取博文的tfidf特征并训练xgboost进行分类预测

——- feat.py：加上一列去掉停用词的微博文本recontent

——- w2c_train_share_resource.py：训练word2vec得到词向量

——- sum_w2c.py: 对用户的词向量求和，生成w2v_sentence(1).csv

——- newmerge_feat.py：在merge_data.csv和stack_new.csv基础上加了一些特征，用到这两个文件

——— source_frequent_count.py：统计各类别source频率，feat是feature_v1的基础上加多了一列    recontent，把那个feature_v1的content中的stopwords去掉，然后统计是主要用recontent

——- word_frequent_count.py：统计各类别source频率，feat是feature_v1的基础上加多了一列recontent，把那个feature_v1的content中的stopwords去掉，然后统计是主要用recontent

——- w2v_prob.py：通过w2v向量预测各类别概率作为特征，主要使用到w2v_sentence1.txt文件，newmerge_feat.csv只是提供uid

——- smp_merge_data.py : 根据原始数据合并数据集

——- smp_process_feature.py ：根据合并的数据集和高频词生成数据特征文件

——- age_predict.py: 得到age的预测结果

——— sex_predict.py: 得到sex的预测结果

——— location_predict.py: 得到location的预测结果

——- merge_submission.py: 合并得到最终的预测结果
