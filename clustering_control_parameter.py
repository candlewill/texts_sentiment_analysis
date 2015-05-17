__author__ = 'NLP-PC'
from parameters import parameters as param

parameters={}
# change to clustering training data
if param['clustering_training_data'] == True:
    from km_cluster import clustering_tweets as km, linear_split as rand_cluster, clustering_tweets_hc as hc, \
        nearest_tweets_cluster as greedy

    parameters['training_clustering_method'] = [km, rand_cluster, hc, greedy][0]
    parameters['num_training_cluster'] = 1000
    # parameters['min_df'] = 1 / 300
    # parameters['TF_binary'] = False

# change to clustering test data
if param['clustering_test_data'] == True:
    from km_cluster import build_clustered_testdata as km, build_clustered_testdata_hc as hc, \
        build_clustered_testdata_nearest as greedy, clustering_texts_using_trainingset as greedy_multi

    parameters['clustering_test_data_method'] = [km, hc, greedy, greedy_multi][0]
    parameters['use_additional_texts'] = (True if parameters['clustering_test_data_method'] == greedy_multi else False)
    if parameters['use_additional_texts'] == False:
        parameters['num_test_cluster'] = 200
    else:
        parameters['cluster_size'] = 10
        parameters['additional_texts']=['training_data', 'test_data'][1]