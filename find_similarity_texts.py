__author__ = 'NLP-PC'
import csv, os, scipy as sp, numpy as np
import sys,math
from sklearn.feature_extraction.text import CountVectorizer
import customed_vectorizer as cst_vectorizer
from sklearn.cluster import KMeans


dir, f = '.\data', 'smilarity_cluster_testdata.csv'
posts = []
with open(os.path.join(dir, f), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    for post in reader:
        posts.extend(post)

#vectorizer = CountVectorizer(min_df=1, stop_words='english')  # min_df决定了CountVectorizer如何处理那些不经常使用的词语（最小文档频率），
# 如果是整数，小于这个值的词语都将被扔掉，如果是一个比例，所有在整个数据集中出现比例小于这个值的词语都将被丢掉
# 使用停用词stop_words，通过'english'或者None的取值可以控制是否使用

# 更改vectorizer，使用StemmedCountVectorizer
# vectorizer=cst_vectorizer.StemmedCountVectorizer(min_df=1,stop_words='english')

# 更改vectorizer, 使用StemmedTfidfVectorizer
vectorizer=cst_vectorizer.StemmedTfidfVectorizer(min_df=1,stop_words='english',decode_error="ignore") # 新版本中用此代替旧版本的charset_error

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape

#K均值聚类
num_clusters=5
km=KMeans(n_clusters=num_clusters,init='random',n_init=1,verbose=1)
km.fit(X_train)
print(km.labels_,km.labels_.shape, num_samples)


print(vectorizer.get_feature_names(), num_features, num_samples)

new_post = "the underlying caste system in america . it's a scathing portrayal"
new_post_vec = vectorizer.transform([new_post])

new_post_label=km.predict(new_post_vec)[0]
# 寻找km.labels_中值等于new_post_label对应的下标，nonzero()作用于这个bool型数组，将这个数组转化为一个更小的数组，它包含True元素对应的索引(index)
similar_indices=(km.labels_==new_post_label).nonzero()[0]
print(similar_indices)
similar=[]
for i in similar_indices:
    dist=sp.linalg.norm((new_post_vec-X_train[i]).toarray())
    similar.append((dist,posts[i]))
similar=sorted(similar)
print(similar)

# 计算向量v1和v2之间的欧式距离（最简单的朴素方法），使用方法：print(dist_raw(vectorizer.transform(['sentence1 one']),vectorizer.transform(['sentence two'])))
def dist_raw(v1, v2):
    # 扩展：使用归一化后的向量
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v1.toarray())

    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())


best_doc = None
best_dist = sys.maxsize  # sys.maxint在新版本中已经取消，由前者代替使用
best_i = None
for i in range(0, num_samples):
    post = posts[i]

    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)

    print('===Post %i with dist=%.2f: %s' % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print('Best post is %i with distance=%.2f, the content is: %s' % (best_i, best_dist, posts[best_i].split(',')[1]))

#查看停用词是什么
print(sorted(vectorizer.get_stop_words()))