__author__ = 'NLP-PC'
__author__ = 'NLP-PC'
import csv, nltk, numpy as np, re, const_values as const
# import vectorizer_estimator as vec_est
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import precision_recall_curve, auc
# from sklearn.cross_validation import ShuffleSplit
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import f1_score
# from sklearn.pipeline import FeatureUnion
# import customed_vectorizer as cstv
# import pylab as pl
# import pickle
# from km_cluster import clustering_tweets


# 预处理
def preprocessor(tweet):
    emo_repl_order = const.emo_repl_order
    emo_repl = const.emo_repl
    re_repl = const.re_repl

    tweet = tweet.lower()
    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])
    tweet = tweet.replace("-", " ").replace("_", " ").replace('"', '').replace(".", '').replace(',', '').replace(';',
                                                                                                                 '').strip()
    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)

    # stem操做
    # english_stemmer=nltk.stem.SnowballStemmer('english')
    # tweet_list=nltk.word_tokenize(tweet)
    # tweet=' '.join([english_stemmer.stem(t) for t in tweet_list])

    return tweet

def load_test_data(classify_type=None):
    file_name='data/raw/testdata.manual.2009.06.14.csv'
    raw_data=csv.reader(open(file_name, 'rt', encoding='utf8'), delimiter=',',quotechar='"')
    tweets,sentiment=[],[]
    for line in raw_data:
        if line[0]=='0' or line[0]=='4':
            tweets.append(line[5])
            sentiment.append(0 if line[0]=='0' else 1)
    return tweets,sentiment
# X,Y=load_test_data()
# print(Y.count(1))
# print(len(X))