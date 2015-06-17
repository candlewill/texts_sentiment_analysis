__author__ = 'NLP-PC'
import numpy as np
import customed_vectorizer as cst_vectorizer
from parameters import vectorizer_param as param

def get_candidate():

    expanding_pos_content = [
        '@williamjone go buy some! you will love me for it!! they are simple yet amazing-ness all rolled into one @drdisaia Just for your comment earlier about the blonde implants ... I like the feedback they are awesome when they are warm. Spesh on a cool day, with a nice strong latte Hey Twitter Im new to this but ive seen my friends do it  Plz Follow me n I will follow u back &lt;3 @randomsonggirl  it is  easier  to tweet if you have  tweetdeck or  seesmic desktop @Jintanut they are awesome when they are warm. Spesh on a cool day, with a nice strong latte. war driving at its best on the way homee, bamboozle was siiicckkk']
    expanding_neg_content = [
        '@stephenkruiser So sorry to hear about your dog I have been accused of being a biscuit fascist because I said Viennese biscuits were not working class All my tweets are already gone, are not they  Missed you guys tonight. Just found out an outbuilding at one of my other houses has been broken into. Again. That is probably the 6th or 7th time now and its done now i need to curl it omg i need someone to do this for me  then my make-up then get dressed then tidy up ... all befor one ! @thecoveted Oooh! I love those earrings! Do you mind if I ask which craft store you went to? We do not have much up here unfortunately']

    return expanding_pos_content, expanding_neg_content

def get_candidate_dynamic(texts, trainingset, cluster_size, file_name):
    vectorizer = cst_vectorizer.StemmedTfidfVectorizer(**param)
    texts_vec = vectorizer.fit_transform(texts)
    training_vec = vectorizer.transform(trainingset)
    from sklearn.metrics.pairwise import pairwise_distances
    # sim_matrix(i, j) is the distance between the ith array from X and the jth array from Y.
    # From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
    sim_matrix = 1 - pairwise_distances(texts_vec, training_vec, metric="cosine")  # euclidean as well
    num_texts = texts_vec.shape[0]
    cluster_size = cluster_size - 1  #减1是因为最后要把texts中放入，所以其实只需选择cluster_size-1个文本
    ind_clustered_tweets = np.zeros([num_texts, cluster_size], dtype=int)

    for i in range(0, num_texts):
        indx = np.argpartition(sim_matrix[i], -cluster_size)[-cluster_size:]
        ind_clustered_tweets[i] = indx

    trainingset = np.array(trainingset)
    clustered_texts = []
    extantion_content = []
    for i in range(0, num_texts):
        ind = ind_clustered_tweets[i]
        clustered_texts.append(texts[i] + ' ' + ' '.join(trainingset[ind]))
        extantion_content.append(' '.join(trainingset[ind]))

    import pickle
    # 推荐file_name的值为neg和pos
    print('和training_data聚合在一起的test data保存在了：./data/extended_test/文件夹*.p中')
    pickle.dump(clustered_texts, open("./data/extended_test_data/" + file_name+"_clustered_texts.p", "wb"))
    pickle.dump(extantion_content, open("./data/extended_test_data/" + file_name+"_extantion_content.p", "wb"))
# 执行上述函数时候需要三种变量