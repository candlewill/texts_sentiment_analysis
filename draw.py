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
predict_label_1 = [0 if rate[i] < 1 else 1 for i in range(0, length)]

# 方法二
predict_label=[]
for i in range(0, length):
    if diff_pos[i]>0 and diff_neg[i]>=0:
        predict_label.append(1)
    elif diff_neg[i]<0 and diff_pos[i]<=0:
        predict_label.append(0)
        print(i)
    else:
        predict_label.append(0 if rate[i] < 1 else 1)
print(predict_label)
print(predict_label_1)

pickle.dump(predict_label, open("./acc_tmp/predict/predict_label.p", "wb"))
import matplotlib.pyplot as plt

# 这样的both值不存在才正常，如果存在需要单独处理
# both = [i if (predict_expanded_texts_with_pos[i] >= predict_expanding_pos_content and predict_expanded_texts_with_neg[i] <= predict_expanding_neg_content) else 0 for i in range(0,length)]
predict_without_clustering =  pickle.load(open("./acc_tmp/predict/predict_without_clustering.p", "rb"))
x_range = np.arange(0, length)
plt.subplot(2, 1, 1)
plt.plot(x_range, predict_expanded_texts_with_pos, 'ro', label='expanded with pos')
plt.plot(x_range, predict_expanded_texts_with_neg, 'bo', label='expanded with neg')
plt.plot(x_range, predict_expanding_pos_content * np.array([1] * length), 'r-', label='pos')
plt.plot(x_range, predict_expanding_neg_content * np.array([1] * length), 'b-', label='neg')
plt.plot(x_range, 0.5 * np.array([1] * length), 'c-', label='0.5')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(x_range, predict_label, 'g.', label='predict labels')
plt.plot(x_range, np.array(predict_without_clustering)*0.8+0.1, 'b.', label='predict_without_clustering')
plt.legend(loc='upper right')
plt.ylim(-0.5, 1.5)
plt.show()
