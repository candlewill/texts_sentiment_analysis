# coding: utf-8
__author__ = 'NLP-PC'
# 分析由predict_using_pretrained_model产生的四个文件中的数据，以函数形式供调用

def analysis_dynamics(expand_with_pos, expand_with_neg, pos_expand_content, neg_expand_content):
    from matplotlib import pyplot as plt

    global g
    plt.scatter(pos_expand_content, neg_expand_content, **g)
    plt.xlabel('pos')
    plt.ylabel('neg')
    plt.show()


import pickle
from Utils import load_test_data

_, true = load_test_data()
length = len(true)
colors = ['green' if true[i] == 1 else 'red' for i in range(0, length)]
g = {'c': colors, 'alpha': 0.7}
expand_with_pos = pickle.load(open("./data/predict_dynamics/predict_pos_clustered_texts_vec.p", "rb"))
expand_with_neg = pickle.load(open("./data/predict_dynamics/predict_neg_clustered_texts_vec.p", "rb"))
pos_expand_content = pickle.load(open("./data/predict_dynamics/predict_pos_extantion_content_vec.p", "rb"))
neg_expand_content = pickle.load(open("./data/predict_dynamics/predict_neg_extantion_content_vec.p", "rb"))
analysis_dynamics(expand_with_pos, expand_with_neg, pos_expand_content, neg_expand_content)
