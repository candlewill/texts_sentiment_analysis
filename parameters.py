__author__ = 'NLP-PC'
# 配置文件

# baseline classifier
parameters = {

    # vectorizer参数选择
    'min_df': 1 / 2000,  # 仅考虑频率出现在min_df之上的ngrams
    'ngram_range': (1, 3),  # ngram范围
    'test_data_size': 10000,  # 选择不同训练数据大小
    'max_df': 0.8,  # 除去太频繁出现的ngrams
    'TF_binary': True,  # 是否使用TF-IDF加权
    'norm': 'l1',  # 是否规格化
    'sublinear_tf': True,  # 是否对TF使用log(1+x)

    # feature type
    'combine_feature': False,  # 是否使用更多的特征

    # 分类器
    'classifier': 'nb',  # 贝叶斯或者svm分类器，目前svm还有问题

    # 是否对training_data分群
    'clustering_training_data': False,  # 具体参数设置在后面的if中
    'num_training_cluster': 0,

    # 是否对test_data分群
    'clustering_test_data': True,
    'num_test_cluster': 0

}

# change to clustering training data
if parameters['clustering_training_data'] == True:
    from km_cluster import clustering_tweets as km, linear_split as rand_cluster, clustering_tweets_hc as hc, \
        nearest_tweets_cluster as greedy

    parameters['training_clustering_method'] = [km, rand_cluster, hc, greedy][3]
    parameters['num_training_cluster'] = 500
    # parameters['min_df'] = 1 / 300
    # parameters['TF_binary'] = False

# change to clustering test data
if parameters['clustering_test_data'] == True:
    from km_cluster import build_clustered_testdata as km, build_clustered_testdata_hc as hc, \
        build_clustered_testdata_nearest as greedy, clustering_texts_using_trainingset as greedy_multi

    parameters['clustering_test_data_method'] = [km, hc, greedy, greedy_multi][0]
    parameters['num_test_cluster'] = 200
    parameters['cluster_size'] = 10