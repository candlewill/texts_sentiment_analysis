__author__ = 'NLP-PC'
import collections, os, csv, numpy as np
import nltk,re
from sklearn.base import BaseEstimator
import pickle


# 使用SentiWordNet返回词语的正负情感得分，函数load_sent_word_net()用于返回一个字典，字典的键（key）是"word type/word"形式的字符串，如"n/implant"，而值(values)是正向和负向分值，简单地对所有同义词的分数求平均值
def load_sent_word_net():
    sent_scores = collections.defaultdict(list)

    data_dir = '.\data'
    with open(os.path.join(data_dir, "SentiWordNet_3.0.0_20130122.txt"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue

            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            # 打印出POS, ID, PosScore, NegScore, SynsetTerms
            for term in SynsetTerms.split(" "):
                # 扔掉每个词后面的数字
                term = term.split("#")[0]
                term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                sent_scores[key].append((float(PosScore), float(NegScore)))
    for key, value in sent_scores.items():
        sent_scores[key] = np.mean(value, axis=0)
    return sent_scores

# 第一次运行时才需要执行，以后就不要执行了可以加快速度
# sent_word_net = load_sent_word_net()
# pickle.dump(sent_word_net, open("./acc_tmp/sent_word_net.p", "wb"))

# 继承自BaseEstimator，需要实现三个方法：get_feature_names(), fit(document), transform(documents)
# 创建自定义vectorizer
class LinguisticVectorizer(BaseEstimator):
    # 返回特征名称列表（list），包含用transform()返回的所有的特征
    def get_feature_names(self):
        return np.array(
            ['sent_neut', 'sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs', 'allcaps', 'exclamation',
             'question', 'hashtag', 'mentioning']
        )

    # As we are not implementing a classifier, we can ignore this one and simply return self.
    # 我们并不进行拟合，但需要返回一个引用
    # 以便可以按照fit(d).transform(d)的方式使用
    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        sent = tuple(d.split())
        tagged = nltk.pos_tag(sent)

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.
        sent_word_net = pickle.load(open("./acc_tmp/sent_word_net.p", "rb"))

        for w, t in tagged:
            p, n = 0, 0
            sent_pos_type = None
            # 名词单词数量
            if t.startswith('NN'):
                sent_pos_type = 'n'
                nouns += 1
            # 形容词
            elif t.startswith('JJ'):
                sent_pos_type = 'a'
                adjectives += 1
            # 动词
            elif t.startswith('VB'):
                sent_pos_type = 'v'
                verbs += 1
            # 副词
            elif t.startswith('RB'):
                sent_pos_type = 'r'
                adverbs += 1

            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, w)

                if sent_word in sent_word_net:
                    p, n = sent_word_net[sent_word]

            pos_vals.append(p)
            neg_vals.append(n)

        l = len(sent)
        # 平均正向得分
        avg_pos_val = np.mean(pos_vals)
        # 平均负向得分
        avg_neg_val = np.mean(neg_vals)
        return [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l, adverbs / l]

    # This returns numpy.array(), containing an array of shape (len(documents), len(get_feature_names)).
    # This means that for every document in documents, it has to return a value for every feature name in get_feature_names().
    def transform(self, documents):
        obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array(
            [self._get_sentiments(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []
        hashtag = []
        mentioning = []

        for d in documents:
            allcaps.append(np.sum([t.isupper() for t in d.split() if len(t) > 2]))
            exclamation.append(d.count("!"))
            question.append(d.count("?"))
            hashtag.append(d.count("#"))
            mentioning.append(d.count("@"))

        result = np.array(
            [obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs, allcaps, exclamation, question, hashtag,
             mentioning]).T
        return result


class StatisticVectorizer(BaseEstimator):
    # 返回特征名称列表（list），包含用transform()返回的所有的特征
    def get_feature_names(self):
        return np.array(
            ['nouns', 'verbs', 'adverbs', 'adjectives', 'allcaps', 'exclamation', 'question', 'hashtag', 'mentioning', 'elongated_words','tweets_len']
        )

    # As we are not implementing a classifier, we can ignore this one and simply return self.
    # 我们并不进行拟合，但需要返回一个引用
    # 以便可以按照fit(d).transform(d)的方式使用
    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        sent = tuple(d.split())
        tagged = nltk.pos_tag(sent)

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        for w, t in tagged:
            # 名词单词数量
            if t.startswith('NN'):
                nouns += 1
            # 形容词
            elif t.startswith('JJ'):
                adjectives += 1
            # 动词
            elif t.startswith('VB'):
                verbs += 1
            # 副词
            elif t.startswith('RB'):
                adverbs += 1

        l = len(sent)
        return [nouns / l , adjectives / l, verbs / l, adverbs/l, l]

    # This returns numpy.array(), containing an array of shape (len(documents), len(get_feature_names)).
    # This means that for every document in documents, it has to return a value for every feature name in get_feature_names().
    def transform(self, documents):
        nouns, adjectives, verbs, adverbs, tweets_len = np.array(
            [self._get_sentiments(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []
        hashtag = []
        mentioning = []
        elongated_words=[]

        for d in documents:
            length=len(d)
            allcaps.append(np.sum([t.isupper() for t in d.split() if len(t) > 2])/length)
            exclamation.append(d.count("!")/length)
            question.append(d.count("?")/length)
            hashtag.append(d.count("#")/length)
            mentioning.append(d.count("@")/length)
            elongated_words.append(len(re.findall(r"([a-zA-Z])\1{2,}",d))/length)

        result = np.array(
            [nouns, adjectives, verbs, adverbs,tweets_len/140, allcaps, exclamation, question, hashtag,
             mentioning, elongated_words]).T
        return np.log1p(result)