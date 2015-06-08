__author__ = 'NLP-PC'
import pickle
import numpy as np

def analysis_result(predict, true):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_recall_fscore_support
    f1 = f1_score(true, predict, average='binary')
    precision_binary, recall_binary, fbeta_score_binary, _ = precision_recall_fscore_support(true, predict, average='binary')
    accuracy = accuracy_score(true, predict)
    print('正确率(Accuracy)：%.3f\nF值(Macro-F score)：%.3f' % (accuracy, f1))
    print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f' % (precision_binary, recall_binary, fbeta_score_binary))

    # 画图
    from matplotlib import pyplot as plt
    n_groups =  5
    values = (accuracy, f1, precision_binary, recall_binary, fbeta_score_binary)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width=0.35
    rects1 = plt.bar(index+ bar_width/2, values, bar_width,alpha=0.6, color='b')
    plt.xlabel('Result')
    plt.ylabel('Scores')
    plt.title('Experiment analysis')
    plt.xticks(index + bar_width, ('Accuracy', 'F', 'Precision', 'Recall', 'F'))
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()


# TRUE LABELS
from Utils import load_test_data
_, true = load_test_data()
# Predict labels
predict = pickle.load(open("./acc_tmp/predict/predict_label.p", "rb"))
analysis_result(predict, true)

predict_without_clustering =  pickle.load(open("./acc_tmp/predict/predict_without_clustering.p", "rb"))
analysis_result(predict_without_clustering, true)