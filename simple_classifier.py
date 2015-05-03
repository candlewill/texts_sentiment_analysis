__author__ = 'NLP-PC'
import csv, numpy as np
import vectorizer_estimator as vec_est
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
import customed_vectorizer as cstv
import pylab as pl
import pickle
from Utils import preprocessor, load_test_data
from km_cluster import clustering_tweets, linear_split, build_clustered_testdata, sentiment_map_cluster2tweets, \
    clustering_tweets_hc, build_clustered_testdata_hc, nearest_tweets_cluster
from sklearn.metrics import precision_recall_fscore_support
import time as time
from parameters import parameters

# 加载数据
def load_data(datatype):
    print('start loading data...')
    # 格式："4","1467822272","Mon Apr 06 22:22:45 PDT 2009","NO_QUERY","ersle","I LOVE @Health4UandPets u guys r the best!! "
    if datatype == 'polarity_classification':
        inpTweets = csv.reader(open('data/training.'+str(parameters['test_data_size'])+'.processed.noemoticon.csv', 'rt', encoding='utf8'),
                               delimiter=',')
        X = []
        Y = []
        for row in inpTweets:
            sentiment = (1 if row[0] == '4' else 0)
            tweet = row[5]
            X.append(sentiment)
            Y.append(tweet)
        # end loop
        return Y, X

    # 格式："positive","the rock is destined to be the 21st century's new "" conan "" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal ."
    elif datatype == 'subjectivity_classification':
        inpTweets = csv.reader(open('data/training_neatfile_4.csv', 'rt', encoding='ISO-8859-1'), delimiter=',',
                               quotechar='"')
        X = []
        Y = []
        for row in inpTweets:
            sentiment = (1 if row[0] == 'positive' or row[0] == 'negative' else 0)
            tweet = row[1]
            X.append(sentiment)
            Y.append(tweet)
        # end loop
        return Y, X

    # 格式：kaggle的标记数据，电影品论
    elif datatype == 'kaggle':
        inpTweets = csv.reader(open('data/labeledTrainData.csv', 'rt', encoding='utf8'),
                               delimiter='\t')
        X = []
        Y = []
        for row in inpTweets:
            sentiment = (1 if row[1] == '1' else 0)
            tweet = row[2]
            X.append(sentiment)
            Y.append(tweet)
        # end loop
        return Y, X


    else:
        return None


# 使用Pipeline类，将向量化处理器和分类器结合到一起，其中TfidfVectorizer作用是将原始推文文本转换为Tf-IDF特征值，以便和标签(label)结合后训练分类器
# 返回的Pipeline实例可以用于fit()和predict()，类似于一个分类器作用
def create_ngram_model(params=None):
    print('start create_ngram_model...')
    # 使用stem在preprocess里面，效果不好
    # tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 2), analyzer='word', binary=False)
    # 自定义StemmedTfidfVectorizer，嵌入stem
    tfidf_ngrams = cstv.StemmedTfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 3), analyzer='word',
                                               binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline(steps=[('vect', tfidf_ngrams), ('clf', clf)])
    # print(sorted(tfidf_ngrams.get_stop_words()))

    return pipeline


# 利用Scikit-learn的FeatureUnion类，把TfidfVectorizer和语言特征（LinguisticVectorizer）结合起来，并行处理
def create_union_model(params=None):
    print('start create_union_model...')
    tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 2), analyzer='word', binary=False)

    ling_status = vec_est.StatisticVectorizer()
    all_features = FeatureUnion([('ling', ling_status), ('tfidf', tfidf_ngrams)])

    clf = MultinomialNB()
    pipeline = Pipeline([("all", all_features), ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline


# 不使用KFold，因为他会把数据切分成连续的几折，相反，使用ShuffleSplit，将数据随机打散，但不能保证相同的数据样本不会出现在多个数据折中
# 训练模型，把创建分类器的函数作为参数传入
def train_model(clc_factory, X, Y):
    print('start train_model...')
    # 设置随机状态，来得到确定性的行为
    cv = ShuffleSplit(n=len(X), n_iter=1, test_size=0.0, random_state=0)

    # accuracy
    scores = []
    # AUC
    pr_scores = []
    # F1 score
    f1 = []

    # 暂时先用for吧，仅仅迭代一次
    for train, test in cv:
        # 测试方法一: do not clustering test data and training data
        # '''
        X_train, Y_train = X[train], Y[train]
        X_test, Y_test = load_test_data()
        X_test, Y_test = np.array(X_test), np.array(Y_test)

        # clf = pickle.load(open("./acc_tmp/clf_all_data_noclustering.p", "rb"))
        clf = clc_factory()
        clf.fit(X_train, Y_train)

        pickle.dump(clf, open("./acc_tmp/clf.p", "wb"))

        # accuracy
        test_score = clf.score(X_test, Y_test)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)

        precision, recall, pr_thresholds = precision_recall_curve(Y_test, proba[:, 1])
        # AUC
        aera_uc = auc(recall, precision)
        pr_scores.append(aera_uc)

        #F1_score
        f1.append(f1_score(Y_test, clf.predict(X_test), average='macro'))

        summary = (np.mean(scores), np.mean(pr_scores), np.mean(f1))
        # Area Under Curve （曲线下面的面积）
        print('正确率(Accuracy)：%.3f\nP/R AUC值：%.3f\nF值(Macro-F score)：%.3f' % (summary))

        # 画图
        pl.clf()
        pl.plot(recall, precision, label='Precision-Recall curve')
        pl.xlabel('Recall')
        pl.ylabel('Precision')
        pl.ylim([0.0, 1.05])
        pl.xlim([0.0, 1.0])
        pl.title('Precision-Recall Curve (AUC=%0.3f)' % aera_uc)
        pl.legend(loc="lower left")
        pl.show()
        # '''
        '''
        # 测试方法2, do not use clustering in test data, changed to specified test data
        X_train, Y_train = X[train], Y[train]
        clf = clc_factory()
        clf.fit(X_train, Y_train)

        pickle.dump(clf, open("./acc_tmp/clf_all_data.p", "wb"))

        X_test, Y_test=load_test_data()
        X_test,X_test_labels=build_clustered_testdata(X_test)
        # print(X_test_labels,len(X_test_labels))

        X_test, Y_test=np.array(X_test), np.array(Y_test)

        true_labels=Y_test
        predict_labels=np.array(sentiment_map_cluster2tweets(clf.predict(X_test),X_test_labels))
        precision,recall,fbeta_score,support=precision_recall_fscore_support(true_labels, predict_labels, average='binary')
        print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f'%(precision,recall,fbeta_score))
        # '''

        '''
        # 测试方法3: 使用 Agglomerative clustering test data
        X_train, Y_train = X[train], Y[train]
        clf = clc_factory()
        clf.fit(X_train, Y_train)

        X_test, Y_test=load_test_data()
        X_test,X_test_labels=build_clustered_testdata_hc(X_test)
        print(X_test_labels,len(X_test_labels))

        X_test, Y_test=np.array(X_test), np.array(Y_test)

        true_labels=Y_test
        predict_labels=np.array(sentiment_map_cluster2tweets(clf.predict(X_test),X_test_labels))
        precision,recall,fbeta_score,support=precision_recall_fscore_support(true_labels, predict_labels, average='binary')
        print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f'%(precision,recall,fbeta_score))
        '''


#寻找模型最优时的参数，使用F值
def grid_search_model(clf_factory, X, Y):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices='True', random_state=0)

    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],  #使用unigrams, bigrams, trigrams
                      vect__min_df=[1, 2],  #最小文档频率，不考虑小于次频率的词
                      vect__stop_words=[None, 'english'],  #是否去除停等词汇
                      vect__smooth_idf=[False, True],  #是否拉普拉斯平滑处理
                      vect__use_idf=[False, True],  # 是否使用IDF
                      vect__sublinear_tf=[False, True],  #是否对词频取对数
                      vect__binary=[False, True],  #试验是否要追踪词语出现次数或者只是简单记录词语出现与否
                      clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],  #Lidstone平滑，加一个常数的平滑方式
                      )

    grid_search = GridSearchCV(clf_factory(), param_grid=param_grid, cv=cv, score_func=f1_score, verbose=10)
    grid_search.fit(X, Y)

    return grid_search.best_estimator_



# st = time.time()
#
# X, Y = load_data('polarity_classification')

# neg, pos = [], []
# for i in range(0, len(Y)):
#     if Y[i] == 0:
#         neg.append(X[i])
#     else:
#         pos.append(X[i])
# # #
# elapsed_time = time.time() - st
# print("Elapsed time: %.3fmin" % (elapsed_time / 60))
# num_cluster = 500
# num_pos_cluster = num_cluster
# num_neg_cluster = num_cluster
# #
# clustered_pos = nearest_tweets_cluster(pos, num_pos_cluster)
# clustered_neg = nearest_tweets_cluster(neg, num_neg_cluster)
#
# elapsed_time = time.time() - st
# print("Elapsed time: %.3fmin" % (elapsed_time / 60))
# #
# X, Y = clustered_pos + clustered_neg, [1] * num_pos_cluster + [0] * num_neg_cluster
# elapsed_time = time.time() - st
# print("Elapsed time: %.3fmin" % (elapsed_time / 60))

# 存一下，防止每次从头运行，速度慢
# pickle.dump(X, open("./acc_tmp/aggregated_tweets.p", "wb"))

# X = pickle.load(open("./acc_tmp/aggregated_tweets.p", "rb"))

# Y = np.array(Y)
# X = np.array(X)
# train_model(create_ngram_model, X, Y)
# train_model(create_union_model, X, Y)

# 如有需要再执行，保存处理的推文
# with open('data/processed_tweets.txt', 'wb') as f:
#     for s in processed_tweets:
#         f.write((s + '\n').encode('utf8'))
# elapsed_time = time.time() - st
# print("Elapsed time: %.3fmin" % (elapsed_time / 60))