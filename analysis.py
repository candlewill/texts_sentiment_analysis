__author__ = 'NLP-PC'
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# TRUE LABELS
from Utils import load_test_data

_, true = load_test_data()
# Predict labels
predict = pickle.load(open("./acc_tmp/predict/predict_label.p", "rb"))
f1 = f1_score(true, predict, average='binary')
precision_binary, recall_binary, fbeta_score_binary, _ = precision_recall_fscore_support(true, predict, average='binary')

print('正确率(Accuracy)：%.3f\nF值(Macro-F score)：%.3f' % (accuracy_score(true, predict), f1))
print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f' % (precision_binary, recall_binary, fbeta_score_binary))
