__author__ = 'NLP-PC'
import pickle
import numpy as np
from Utils import load_test_data

text, true = load_test_data()
with open('acc_tmp/text.csv', 'w', newline='') as f:
    for line in text:
        f.write(line+ '\n')

predict_expanding_pos_content = pickle.load(open("./acc_tmp/predict/predict_expanding_pos_content.p", "rb"))
predict_expanding_neg_content = pickle.load(open("./acc_tmp/predict/predict_expanding_neg_content.p", "rb"))
predict_expanded_texts_with_pos = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_pos.p", "rb"))
predict_expanded_texts_with_neg = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_neg.p", "rb"))
length = len(predict_expanded_texts_with_pos)

diff_pos = np.array(predict_expanded_texts_with_pos) - predict_expanding_pos_content
diff_neg = np.array(predict_expanded_texts_with_neg) - predict_expanding_neg_content

ratio_pos = np.array(predict_expanded_texts_with_pos) / predict_expanding_pos_content
ratio_neg = np.array(predict_expanded_texts_with_neg) / predict_expanding_neg_content

diff = np.abs(diff_pos) - np.abs(diff_neg)

# 方法一，比较偏差的多少
predict_label_1 = [1 if diff[l] < 0 else 0 for l in range(0, length)]

# 方法二，部分比较偏差多少
predict_label_2 = []
for i in range(0, length):
    if diff_pos[i] > 0:
        predict_label_2.append(1)
    elif diff_neg[i] < 0:
        predict_label_2.append(0)
    else:
        predict_label_2.append(1 if diff[i] < 0 else 0)

# 方法三，部分使用相对偏差多少
predict_label_3 = predict_label_2.copy()
for j in range(0, length):
    if diff_pos[j] < 0 and diff_neg[j] > 0:
        predict_label_3[j] = (
            1 if diff_pos[j] / predict_expanding_pos_content - diff_neg[j] / predict_expanding_neg_content < 0 else 0)

# 方法四，全部使用相对偏差多少
predict_label_4 = [
    1 if diff_pos[k] / predict_expanding_pos_content - diff_neg[k] / predict_expanding_neg_content < 0 else 0 for k in
    range(0, length)]

pickle.dump(predict_label_1, open("./acc_tmp/predict/predict_label_1.p", "wb"))
pickle.dump(predict_label_2, open("./acc_tmp/predict/predict_label_2.p", "wb"))
pickle.dump(predict_label_3, open("./acc_tmp/predict/predict_label_3.p", "wb"))
pickle.dump(predict_label_4, open("./acc_tmp/predict/predict_label_4.p", "wb"))

# 这样的both值不存在才正常，如果存在需要单独处理
# both = [i if (predict_expanded_texts_with_pos[i] >= predict_expanding_pos_content and predict_expanded_texts_with_neg[i] <= predict_expanding_neg_content) else 0 for i in range(0,length)]
predict_without_clustering = pickle.load(open("./acc_tmp/predict/predict_without_clustering.p", "rb"))

print(predict_label_1)
print(predict_label_2)
print(predict_label_3)
print(predict_label_4)
print(true)
print(predict_without_clustering)
exit()

import matplotlib.pyplot as plt
x_range = np.arange(0, length)
plt.subplot(2, 1, 1)
plt.plot(x_range, predict_expanded_texts_with_pos, 'ro', label='expanded with pos')
plt.plot(x_range, predict_expanded_texts_with_neg, 'bo', label='expanded with neg')
plt.plot(x_range, predict_expanding_pos_content * np.array([1] * length), 'r-', label='pos')
plt.plot(x_range, predict_expanding_neg_content * np.array([1] * length), 'b-', label='neg')
plt.plot(x_range, 0.5 * np.array([1] * length), 'c-', label='0.5')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(x_range, np.array(predict_label_1) * 0.9 + 0.05, 'r.', label='predict labels_1')
plt.plot(x_range, np.array(predict_label_2) * 0.7 + 0.15, 'g.', label='predict labels_2')
plt.plot(x_range, np.array(predict_label_3) * 0.6 + 0.2, 'b.', label='predict labels_3')
plt.plot(x_range, np.array(predict_without_clustering) * 0.8 + 0.1, 'c.', label='predict_without_clustering')
plt.legend(loc='upper right')
plt.ylim(-0.5, 1.5)
plt.show()

# 下面三个图片共用的参数
colors = ['green' if true[i] == 1 else 'red' for i in range(0, length)]
area = 15  # 0 to 15 point radiuses
labels = ['T{0}'.format(i) for i in range(length)]
# diff_pos, diff_neg
plt.scatter(diff_pos, diff_neg, s=area, c=colors, alpha=0.7)
# add annotation
for label, x, y in zip(labels, diff_pos, diff_neg):
    plt.annotate(label,xy=(x, y))

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('diff pos')
plt.ylabel('diff neg')
plt.show()

# diff/pos
plt.scatter(np.array(diff_pos) / predict_expanding_pos_content, np.array(diff_neg) / predict_expanding_neg_content,
            s=area, c=colors, alpha=0.7)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('diff_pos/pos')
plt.ylabel('diff_neg/neg')
plt.show()

# ratio
plt.scatter(ratio_pos, ratio_neg, s=area, c=colors, alpha=0.7)
plt.xlabel('ratio_pos')
plt.ylabel('ratio_neg')
plt.show()
