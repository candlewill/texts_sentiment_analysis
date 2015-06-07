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
    expanding_pos_content = ['@williamjone go buy some! you will love me for it!! they are simple yet amazing-ness all rolled into one']
    expanding_neg_content = ['@stephenkruiser So sorry to hear about your dog.']
    expanding_pos_content, expanding_neg_content = np.array(expanding_pos_content), np.array(expanding_neg_content)
    expanding_pos_content_vec, expanding_neg_content_vec = vectorizer.transform(expanding_pos_content), vectorizer.transform(expanding_neg_content)

    from test_data_clustering import expand_text_list as expanding_method
    expanded_texts_with_pos, expanded_texts_with_neg = expanding_method(X_train, expanding_pos_content), expanding_method(X_train, expanding_neg_content)
    expanded_texts_vec_with_pos, expanded_texts_vec_with_neg = vectorizer.transform(expanded_texts_with_pos), vectorizer.transform(expanded_texts_with_neg)

    clf = MultinomialNB()
    clf.fit(trian_vec, Y_train)

    predict_expanding_pos_content = clf.predict_proba(expanding_pos_content_vec)[:, 1]
    predict_expanding_neg_content = clf.predict_proba(expanding_neg_content_vec)[:, 1]

    predict_expanded_texts_with_pos, predict_expanded_texts_with_neg = clf.predict_proba(expanded_texts_vec_with_pos)[:, 1], clf.predict_proba(expanded_texts_vec_with_neg)[:, 1]

    print(predict_expanding_pos_content, predict_expanding_neg_content, predict_expanded_texts_with_pos, predict_expanded_texts_with_neg)

    # 保存预测结果
    pickle.dump(predict_expanding_pos_content, open("./acc_tmp/predict_expanding_pos_content.p", "wb"))
    pickle.dump(predict_expanding_neg_content, open("./acc_tmp/predict_expanding_neg_content.p", "wb"))
    pickle.dump(predict_expanded_texts_with_pos, open("./acc_tmp/predict_expanded_texts_with_pos.p", "wb"))
    pickle.dump(predict_expanded_texts_with_neg, open("./acc_tmp/predict_expanded_texts_with_neg.p", "wb"))
    print('变量成功保存在./acc_tmp/ ^_^')
    # 画图


