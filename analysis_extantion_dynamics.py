# coding: utf-8
__author__ = 'NLP-PC'
import pickle
# 分析由predict_using_pretrained_model产生的四个文件中的数据，以函数形式供调用
predict_lable_testdata_without_clustering = pickle.load(open("./data/predict_dynamics/predict_lable_testdata_without_clustering.p", "rb"))


def analysis_dynamics(expand_with_pos, expand_with_neg, pos_expand_content, neg_expand_content):
    from matplotlib import pyplot as plt

    pridict_text = pickle.load(open("./data/predict_dynamics/predict_testdata_without_clustering.p", "rb"))
    global g
    plt.scatter(range(0, length), pridict_text, **g)
    plt.xlabel('n')
    plt.ylabel('prob')
    plt.figure()
    plt.scatter(expand_with_pos, expand_with_neg, **g)
    plt.xlabel('expand_with_pos')
    plt.ylabel('expand_with_neg')
    plt.figure()
    plt.scatter(pos_expand_content, neg_expand_content, **g)
    plt.xlabel('pos_expand_content')
    plt.ylabel('neg_expand_content')
    plt.figure()
    plt.scatter(expand_with_pos - pos_expand_content, expand_with_neg - neg_expand_content, **g)
    plt.xlabel('pos -')
    plt.ylabel('neg -')
    plt.figure()
    plt.scatter(expand_with_pos / pos_expand_content, expand_with_neg / neg_expand_content, **g)
    plt.xlabel('pos /')
    plt.ylabel('neg /')
    plt.figure()
    plt.scatter(expand_with_pos + pos_expand_content, expand_with_neg + neg_expand_content, **g)
    plt.xlabel('pos +')
    plt.ylabel('neg +')
    plt.figure()
    plt.scatter(1 - expand_with_pos, expand_with_neg, **g)
    plt.xlabel('pos distance')
    plt.ylabel('neg distance')
    plt.show()

    plt.plot(range(0, length), expand_with_pos, 'r-', alpha=0.8)
    plt.scatter(range(0,length), expand_with_pos, marker='o', c=colors)
    plt.plot(range(0, length), pos_expand_content, 'r-', alpha=0.4)
    plt.scatter(range(0,length), expand_with_neg, marker='o', c=colors)
    plt.axhline(0.5, color='black')
    plt.plot(range(0, length), expand_with_neg, 'g-', alpha=0.8)
    plt.plot(range(0, length), neg_expand_content, 'g-', alpha=0.4)
    plt.show()

import pickle
from Utils import load_test_data

_, true = load_test_data()
length = len(true)
colors = ['red' if true[i] == 1 else 'green' for i in range(0, length)]
g = {'c': colors, 'alpha': 0.7}
expand_with_pos = pickle.load(open("./data/predict_dynamics/predict_pos_clustered_texts_vec.p", "rb"))
expand_with_neg = pickle.load(open("./data/predict_dynamics/predict_neg_clustered_texts_vec.p", "rb"))
pos_expand_content = pickle.load(open("./data/predict_dynamics/predict_pos_extantion_content_vec.p", "rb"))
neg_expand_content = pickle.load(open("./data/predict_dynamics/predict_neg_extantion_content_vec.p", "rb"))
analysis_dynamics(expand_with_pos, expand_with_neg, pos_expand_content, neg_expand_content)

# predict
predict1 =[0 if ele==True else 1 for ele in ((1 - expand_with_pos) > expand_with_neg)]
from analysis import analysis_result as ar
ar(predict1,true)
ar(predict_lable_testdata_without_clustering,true)