# coding: utf-8
__author__ = 'NLP-PC'
# 由于错误的使用fit_transform, 一下代码已经整合到clustering_fixed_candidate.py中
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 加载测试数据
neg_clustered_texts = pickle.load(open("./data/extended_test_data/neg_clustered_texts.p", "rb"))
neg_extantion_content = pickle.load(open("./data/extended_test_data/neg_extantion_content.p", "rb"))
pos_clustered_texts = pickle.load(open("./data/extended_test_data/pos_clustered_texts.p", "rb"))
pos_extantion_content = pickle.load(open("./data/extended_test_data/pos_extantion_content.p", "rb"))
logger.info('加载测试数据完成')

# 加载模型
clf = pickle.load(open("./acc_tmp/predict/classifier.p", "rb"))
logger.info('成功加载分类器模型')
# 向量化
from customed_vectorizer import StemmedTfidfVectorizer
from parameters import vectorizer_param as param
vectorizer = StemmedTfidfVectorizer(**param)
neg_clustered_texts_vec = vectorizer.fit_transform(neg_clustered_texts)
neg_extantion_content_vec = vectorizer.fit_transform(neg_extantion_content)
pos_clustered_texts_vec = vectorizer.fit_transform(pos_clustered_texts)
pos_extantion_content_vec = vectorizer.fit_transform(pos_extantion_content)
logger.info('向量化完成')

# 预测
predict_neg_clustered_texts_vec = clf.predict_proba(neg_clustered_texts_vec)[:, 1]
predict_neg_extantion_content_vec = clf.predict_proba(neg_extantion_content_vec)[:, 1]
predict_pos_clustered_texts_vec = clf.predict_proba(pos_clustered_texts_vec)[:, 1]
predict_pos_extantion_content_vec = clf.predict_proba(pos_extantion_content_vec)[:, 1]
from Utils import load_test_data

text, _ = load_test_data()
predict_testdata_without_clustering = clf.predict_proba(vectorizer.fit_transform(text))[:, 1]
logger.info('完成预测，即将保存')

# 保存结果
pickle.dump(predict_neg_clustered_texts_vec, open("./data/predict_dynamics/predict_neg_clustered_texts_vec.p", "wb"))
pickle.dump(predict_neg_extantion_content_vec, open("./data/predict_dynamics/predict_neg_extantion_content_vec.p", "wb"))
pickle.dump(predict_pos_clustered_texts_vec, open("./data/predict_dynamics/predict_pos_clustered_texts_vec.p", "wb"))
pickle.dump(predict_pos_extantion_content_vec, open("./data/predict_dynamics/predict_pos_extantion_content_vec.p", "wb"))
pickle.dump(predict_testdata_without_clustering, open("./data/predict_dynamics/predict_testdata_without_clustering.p", "wb"))
logger.info('完成保存')