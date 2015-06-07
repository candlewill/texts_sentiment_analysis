__author__ = 'NLP-PC'
import pickle
import numpy as np

predict_expanding_pos_content = pickle.load(open("./acc_tmp/predict/predict_expanding_pos_content.p", "rb"))
predict_expanding_neg_content = pickle.load(open("./acc_tmp/predict/predict_expanding_neg_content.p", "rb"))
predict_expanded_texts_with_pos = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_pos.p", "rb"))
predict_expanded_texts_with_neg = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_neg.p", "rb"))
length = len(predict_expanded_texts_with_pos)
print(predict_expanded_texts_with_pos)

diff_pos = np.array(predict_expanded_texts_with_pos) - predict_expanding_pos_content
diff_neg = np.array(predict_expanded_texts_with_neg) - predict_expanding_neg_content

rate = np.abs(np.divide(diff_neg, diff_pos))
# 方法一，求比值
predict_label = [0 if rate[i]<1 else 1 for i in range(0, length)]
pickle.dump(predict_label, open("./acc_tmp/predict/predict_label.p", "wb"))
import matplotlib.pyplot as plt

x_range = np.arange(0, length)
plt.subplot(2, 1, 1)
plt.plot(x_range, predict_expanded_texts_with_pos, 'ro', label='expanded with pos')
plt.plot(x_range, predict_expanded_texts_with_neg, 'bo', label='expanded with neg')
plt.plot(x_range, predict_expanding_pos_content * np.array([1] * length), 'r-', label='pos')
plt.plot(x_range, predict_expanding_neg_content * np.array([1] * length), 'b-', label='neg')
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.plot(x_range, predict_label, 'g.', label = 'predict labels')
plt.legend(loc='upper right')
plt.ylim(-0.5,1.5)
plt.show()