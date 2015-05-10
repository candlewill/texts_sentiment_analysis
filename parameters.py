__author__ = 'NLP-PC'
# 配置文件

# baseline classifier
parameters = {

    # vectorizer参数选择
    'min_df': 1/2000,  # 仅考虑频率出现在min_df之上的ngrams
    'ngram_range': (1, 3),  # ngram范围
    'test_data_size': 10000,  # 选择不同训练数据大小
    'max_df': 0.8,  # 除去太频繁出现的ngrams
    'TF_binary': True,  # 是否使用TF-IDF加权
    'norm': 'l1',  # 是否规格化
    'sublinear_tf': True,  # 是否对TF使用log(1+x)

    # feature type
    'combine_feature':False, # 是否使用更多的特征

    # 分类器
    'classifier': 'nb',  # 贝叶斯或者svm分类器，目前svm还有问题

    # 是否对training_data分群
    'clustering_training_data': True,
    'num_training_cluster': 500,

    # 是否对test_data分群
    'clustering_test_data': False,
    'num_test_cluster': 0

}

# change to clustering training data
if parameters['clustering_training_data']==True:
    parameters['num_training_cluster']=500
    parameters['min_df']=1/300
    parameters['TF_binary']=False

# change to clustering test data
if parameters['clustering_test_data']==True:
    parameters['num_test_cluster']=250