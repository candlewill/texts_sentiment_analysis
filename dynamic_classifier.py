#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'NLP-PC'
from simple_classifier import load_data
import pickle
import numpy as np
import pylab as pl
from parameters import parameters
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
from clustering_control_parameter import parameters as clustering_control_param

X, Y = load_data('polarity_classification')
X, Y = np.array(X), np.array(Y)

from sklearn.cross_validation import ShuffleSplit

cv = ShuffleSplit(n=len(X), n_iter=1, test_size=0.0, random_state=0)

from Utils import preprocessor
from customed_vectorizer import StemmedTfidfVectorizer
from parameters import vectorizer_param as param

vectorizer = StemmedTfidfVectorizer(**param)

from sklearn.naive_bayes import MultinomialNB

for train, _ in cv:

    X_train, Y_train = X[train], Y[train]
    neg, pos = [], []
    for i in range(0, len(Y_train)):
        if Y_train[i] == 0:
            neg.append(X_train[i])
        else:
            pos.append(X_train[i])
    num_cluster = clustering_control_param['num_training_cluster']
    num_pos_cluster = num_cluster
    num_neg_cluster = num_cluster
    clustering_testdata = clustering_control_param['training_clustering_method']
    clustered_pos = clustering_testdata(pos, num_pos_cluster)
    clustered_neg = clustering_testdata(neg, num_neg_cluster)
    X_train, Y_train = clustered_pos + clustered_neg, [1] * num_pos_cluster + [0] * num_neg_cluster
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    trian_vec = vectorizer.fit_transform(X_train)

    # clustering test data
    from candidate_content import get_candidate

    expanding_pos_content, expanding_neg_content = get_candidate()
    expanding_pos_content, expanding_neg_content = np.array(expanding_pos_content), np.array(expanding_neg_content)
    expanding_pos_content_vec, expanding_neg_content_vec = vectorizer.transform(
        expanding_pos_content), vectorizer.transform(expanding_neg_content)

    # 加载测试资料
    from Utils import load_test_data

    X_test, Y_test = load_test_data()

    # 下面代码临时执行，需要用时再执行，不用时注释，用来产生扩展的test data
    # from candidate_content import get_candidate_dynamic
    # get_candidate_dynamic(X_test, neg, 5, 'neg')
    # get_candidate_dynamic(X_test, pos, 5, 'pos')
    # exit()
    # 临时执行代码结束，为了更好的组织代码结构，将此文件另存为一份dynamic_classifer.py

