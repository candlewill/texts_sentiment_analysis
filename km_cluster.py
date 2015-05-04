__author__ = 'NLP-PC'
import csv, os, scipy as sp, numpy as np
import sys, math
from sklearn.feature_extraction.text import CountVectorizer
import customed_vectorizer as cst_vectorizer
from sklearn.cluster import KMeans
from Utils import preprocessor as preprocessor
from random import shuffle

# 格式： tweet集合list
def clustering_tweets(labeled_tweets, num_cluster):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(labeled_tweets)
    km = KMeans(n_clusters=num_cluster, init='k-means++', n_init=1, verbose=1)
    km.fit(tweet_vec)
    clustered_tweets = []
    for i in range(0, num_cluster):
        similar_indices = (km.labels_ == i).nonzero()[0]
        sent = ''
        for sid in similar_indices:
            sent = labeled_tweets[sid] + ' ' + sent
        clustered_tweets.append(sent)
    return clustered_tweets


# test for clustering_tweets function
# dir, f = '.\data', 'smilarity_cluster_testdata.csv'
# posts = []
# with open(os.path.join(dir, f), 'r') as csvfile:
# reader = csv.reader(csvfile, delimiter=',', quotechar='"')
#     for post in reader:
#         posts.append(post[1])
# print(posts)
# print(len(posts))
# print(clustering_tweets(posts,3))

def linear_split(labeled_tweets, num_cluster):
    shuffle(labeled_tweets)  #是否shuffle，视情况而定
    counter = 0
    num_within_cluster = math.ceil(len(labeled_tweets) / num_cluster)
    clustered_tweets = []
    sent = ''
    for line in labeled_tweets:
        sent = str(line + ' ' + sent)
        counter += 1
        if counter >= num_within_cluster:  # 有点问题
            clustered_tweets.append(sent)
            counter = 0
            sent = ''

        else:
            continue
    return clustered_tweets


# test for linear_split function
# dir, f = '.\data', 'smilarity_cluster_testdata.csv'
# posts = []
# with open(os.path.join(dir, f), 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='"')
#     for post in reader:
#         posts.append(post[1])
# print(posts)
# print(len(posts))
# print(linear_split(posts,3))

def build_clustered_testdata(tweets):
    num_cluster = 240
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(tweets)
    km = KMeans(n_clusters=num_cluster, init='k-means++', n_init=30, verbose=1)
    km.fit(tweet_vec)
    clustered_tweets = []
    for i in range(0, num_cluster):
        similar_indices = (km.labels_ == i).nonzero()[0]
        sent = ''
        for sid in similar_indices:
            sent = tweets[sid] + ' ' + sent
        clustered_tweets.append(sent)
    return clustered_tweets, km.labels_


def sentiment_map_cluster2tweets(cluster_senti, original_labels):
    tweets_senti = []
    for i in original_labels:
        tweets_senti.append(cluster_senti[i])
    return tweets_senti


# Test sentiment_map_cluster2tweets
# cluster_senti=[0,1,0]
# original_labels=[0,1,0,0,1,2,1,2]
# print(sentiment_map_cluster2tweets(cluster_senti, original_labels))

def clustering_tweets_hc(labeled_tweets, num_cluster):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(labeled_tweets).toarray()
    # print(tweet_vec)
    n_clusters = num_cluster

    from sklearn.neighbors import kneighbors_graph

    knn_graph = kneighbors_graph(tweet_vec, 1, include_self=False)
    # print(knn_graph)

    connectivity = knn_graph
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(linkage='ward', connectivity=connectivity, n_clusters=n_clusters)
    model.fit(tweet_vec)
    c = model.labels_
    # print(c,len(c))

    clustered_tweets = []
    for i in range(0, num_cluster):
        similar_indices = (c == i).nonzero()[0]
        sent = ''
        for sid in similar_indices:
            sent = labeled_tweets[sid] + ' ' + sent
        clustered_tweets.append(sent)
    return clustered_tweets
    # test clustering_tweets_hc
    # T=['we are loving each other', 'we are good', 'loving is good', 'go to each other heart', 'nice to meet u']
    # print(T, clustering_tweets_hc(T, 2))

# 只有tweets一个参数，没有n_cluster需要在函数内部指定
def build_clustered_testdata_hc(tweets):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(tweets).toarray()
    # print(tweet_vec)
    n_clusters = 3

    from sklearn.neighbors import kneighbors_graph

    knn_graph = kneighbors_graph(tweet_vec, 2, include_self=False)
    # print(knn_graph)

    connectivity = knn_graph
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(linkage='average', connectivity=connectivity, n_clusters=n_clusters, affinity='cosine')
    model.fit(tweet_vec)
    c = model.labels_
    # print(c,len(c))

    clustered_tweets = []
    for i in range(0, n_clusters):
        similar_indices = (c == i).nonzero()[0]
        sent = ''
        for sid in similar_indices:
            sent = tweets[sid] + ' ' + sent
        clustered_tweets.append(sent)
    return clustered_tweets, c
# Test
# T=['T1 we are loving each other', 'T2 we are good', 'T3 loving is good', 'T4 go to each other heart', 'T5 nice to meet u', 'T6 you are not good']
# clustered_texts,c=build_clustered_testdata_hc(T)
# print(clustered_texts,c)


def nearest_tweets_cluster(tweets, n_clusters):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(tweets)

    from sklearn.metrics.pairwise import pairwise_distances
    sim_matrix = 1 - pairwise_distances(tweet_vec, metric="cosine")  # euclidean as well
    num_tweets=tweet_vec.shape[0]

    num_clusters=n_clusters
    num_tweets_in_cluster=math.ceil(num_tweets/num_clusters)

    ind_clustered_tweets=np.zeros([num_clusters,num_tweets_in_cluster],dtype=int)
    j=0
    for i in range(0,num_tweets):
        if np.any(sim_matrix[i] != -np.inf) and j<num_clusters:
            indx =np.argpartition(sim_matrix[i], -num_tweets_in_cluster)[-num_tweets_in_cluster:]
            ind_clustered_tweets[j] = [ind if sim_matrix[i, ind]!=-np.inf else -1 for ind in indx]
            # ind_clustered_tweets[j]=indx
            sim_matrix[:, indx]=-np.inf
            sim_matrix[indx, :]=-np.inf
            j += 1

        elif j>=num_clusters:
            break
        else: continue

    tweets=np.array(tweets)
    clustered_tweets=[]
    for i in range(0, num_clusters):
        ind=ind_clustered_tweets[i]
        ind_illegal=np.where(ind==-1)[0] # index of -1
        # print(ind_illegal)
        if len(ind_illegal)!=0:
            ind=np.delete(ind, ind_illegal)

        # print(ind)
        clustered_tweets.append(' '.join(tweets[ind]))
    import pickle
    print('聚合在一起的training data保存在了：./acc_tmp/aggregated_training_tweets_greedy.p文件中')
    pickle.dump(clustered_tweets, open("./acc_tmp/aggregated_training_tweets_greedy.p", "wb"))
    return (clustered_tweets)
# T=['T1 we are loving each other', 'T2 we are good', 'T3 loving is good', 'T4 go to each other heart', 'T5 nice to meet u', 'T6 you are not good']
# print(nearest_tweets_cluster(T,3))
# ['T4 go to each other heart T2 we are good T1 we are loving each other', 'T5 nice to meet u T3 loving is good T5 nice to meet u']


# 将nearest_tweets_cluster改造为可以聚类测试数据(需要返回每一个tweets对应的cluster编号)，和nearest_tweets_cluster类似，上面几乎都是这样的，聚类有两个，所以看起来复杂
def build_clustered_testdata_nearest(tweets):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(preprocessor=preprocessor, min_df=1, stop_words=None,
                                                       decode_error="ignore")
    tweet_vec = vectorizer.fit_transform(tweets)

    from sklearn.metrics.pairwise import pairwise_distances
    sim_matrix = 1 - pairwise_distances(tweet_vec, metric="cosine")  # euclidean as well
    num_tweets=tweet_vec.shape[0]

    from parameters import parameters
    num_clusters=parameters['num_test_cluster']
    num_tweets_in_cluster=math.ceil(num_tweets/num_clusters) # 一共100tweets放在21个cluster中就会出错：最后一个cluster为空

    ind_clustered_tweets=np.zeros([num_clusters,num_tweets_in_cluster],dtype=int)
    j=0
    for i in range(0,num_tweets):
        if np.any(sim_matrix[i] != -np.inf) and j<num_clusters:
            indx =np.argpartition(sim_matrix[i], -num_tweets_in_cluster)[-num_tweets_in_cluster:]
            ind_clustered_tweets[j] = [ind if sim_matrix[i, ind]!=-np.inf else -1 for ind in indx]
            # ind_clustered_tweets[j]=indx
            sim_matrix[:, indx]=-np.inf
            sim_matrix[indx, :]=-np.inf
            j += 1

        elif j>=num_clusters:
            break
        else: continue

    tweets=np.array(tweets)
    clustered_tweets=[]
    for i in range(0, num_clusters):
        ind=ind_clustered_tweets[i]
        ind_illegal=np.where(ind==-1)[0] # index of -1
        # print(ind_illegal)
        if len(ind_illegal)!=0:
            ind=np.delete(ind, ind_illegal)

        # print(ind)
        clustered_tweets.append(' '.join(tweets[ind]))
    import pickle
    print('聚合在一起的test data 保存在了：./acc_tmp/aggregated_test_tweets_greedy.p文件中')
    pickle.dump(clustered_tweets, open("./acc_tmp/aggregated_test_tweets_greedy.p", "wb"))
    return (clustered_tweets,[np.where(ind_clustered_tweets==tweets_id)[0][0] for tweets_id in range(0, num_tweets)])
# test
# T=['T1 we are loving each other', 'T2 we are good', 'T3 loving is good', 'T4 go to each other heart', 'T5 nice to meet u', 'T6 you are not good']
# print(build_clustered_testdata_nearest(T))