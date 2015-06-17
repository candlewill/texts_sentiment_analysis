# coding: utf-8
__author__ = 'NLP-PC'
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
neg_clustered_texts_vec = vectorizer.transform(neg_clustered_texts)
neg_extantion_content_vec = vectorizer.transform(neg_extantion_content)
pos_clustered_texts_vec = vectorizer.transform(pos_clustered_texts)
pos_extantion_content_vec = vectorizer.transform(pos_extantion_content)
