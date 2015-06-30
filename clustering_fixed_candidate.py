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
    from candidate_content import get_candidate_dynamic
    # get_candidate_dynamic(X_test, neg, 8, 'neg')
    # get_candidate_dynamic(X_test, pos, 8, 'pos')
    # exit()
    # 临时执行代码结束，为了更好的组织代码结构，将此文件另存为一份dynamic_classifer.py

    from test_data_clustering import expand_text_list as expanding_method

    expanded_texts_with_pos, expanded_texts_with_neg = expanding_method(X_test,
                                                                        expanding_pos_content), expanding_method(X_test,
                                                                                                                 expanding_neg_content)
    expanded_texts_vec_with_pos, expanded_texts_vec_with_neg = vectorizer.transform(
        expanded_texts_with_pos), vectorizer.transform(expanded_texts_with_neg)

    clf = MultinomialNB()
    clf.fit(trian_vec, Y_train)

    # 保存分类器模型
    pickle.dump(clf, open("./acc_tmp/predict/classifier.p", "wb"))
    print('贝叶斯分类器保存在了./acc_tmp/文件夹classifier.p中 OK')

    # 方法一
    # predict_expanding_pos_content = clf.predict_proba(expanding_pos_content_vec)[:, 1]
    # predict_expanding_neg_content = clf.predict_proba(expanding_neg_content_vec)[:, 1]
    #
    # predict_expanded_texts_with_pos, predict_expanded_texts_with_neg = clf.predict_proba(expanded_texts_vec_with_pos)[:,
    #                                                                    1], clf.predict_proba(
    #     expanded_texts_vec_with_neg)[:, 1]
    #
    # predict_without_clustering = clf.predict(vectorizer.transform(X_test))
    # print(predict_expanding_pos_content, predict_expanding_neg_content, predict_expanded_texts_with_pos,
    #       predict_expanded_texts_with_neg)
    #
    # # 保存预测结果
    # pickle.dump(predict_expanding_pos_content, open("./acc_tmp/predict/predict_expanding_pos_content.p", "wb"))
    # pickle.dump(predict_expanding_neg_content, open("./acc_tmp/predict/predict_expanding_neg_content.p", "wb"))
    # pickle.dump(predict_expanded_texts_with_pos, open("./acc_tmp/predict/predict_expanded_texts_with_pos.p", "wb"))
    # pickle.dump(predict_expanded_texts_with_neg, open("./acc_tmp/predict/predict_expanded_texts_with_neg.p", "wb"))
    # pickle.dump(predict_without_clustering, open("./acc_tmp/predict/predict_without_clustering.p", "wb"))
    # print('变量成功保存在./acc_tmp/ ^_^')
    # # 画图
    # 方法一结束


    # 方法二
    # 强制插入
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
    # clf = pickle.load(open("./acc_tmp/predict/classifier.p", "rb"))
    # logger.info('成功加载分类器模型')
    # 向量化
    # from customed_vectorizer import StemmedTfidfVectorizer
    # from parameters import vectorizer_param as param
    # vectorizer = StemmedTfidfVectorizer(**param)
    neg_clustered_texts_vec = vectorizer.transform(neg_clustered_texts)
    neg_extantion_content_vec = vectorizer.transform(neg_extantion_content)
    pos_clustered_texts_vec = vectorizer.transform(pos_clustered_texts)
    pos_extantion_content_vec = vectorizer.transform(pos_extantion_content)
    logger.info('向量化完成')

    # 预测
    predict_neg_clustered_texts_vec = clf.predict_proba(neg_clustered_texts_vec)[:, 1]
    predict_neg_extantion_content_vec = clf.predict_proba(neg_extantion_content_vec)[:, 1]
    predict_pos_clustered_texts_vec = clf.predict_proba(pos_clustered_texts_vec)[:, 1]
    predict_pos_extantion_content_vec = clf.predict_proba(pos_extantion_content_vec)[:, 1]
    from Utils import load_test_data

    text, _ = load_test_data()
    transformed_text = vectorizer.transform(text)
    predict_testdata_without_clustering = clf.predict_proba(transformed_text)[:, 1]
    predict_lable_testdata_without_clustering = clf.predict(transformed_text)
    logger.info('完成预测，即将保存')

    # 保存结果
    pickle.dump(predict_neg_clustered_texts_vec, open("./data/predict_dynamics/predict_neg_clustered_texts_vec.p", "wb"))
    pickle.dump(predict_neg_extantion_content_vec, open("./data/predict_dynamics/predict_neg_extantion_content_vec.p", "wb"))
    pickle.dump(predict_pos_clustered_texts_vec, open("./data/predict_dynamics/predict_pos_clustered_texts_vec.p", "wb"))
    pickle.dump(predict_pos_extantion_content_vec, open("./data/predict_dynamics/predict_pos_extantion_content_vec.p", "wb"))
    pickle.dump(predict_testdata_without_clustering, open("./data/predict_dynamics/predict_testdata_without_clustering.p", "wb"))
    pickle.dump(predict_lable_testdata_without_clustering, open("./data/predict_dynamics/predict_lable_testdata_without_clustering.p", "wb"))
    logger.info('完成保存')
    # 强制插入结束