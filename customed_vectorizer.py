__author__ = 'NLP-PC'
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#利用NLTK进行词干化处理
english_stemmer=nltk.stem.SnowballStemmer('english')

# 在把post传入CountVectorizer之前，需要对他们进行词干处理。虽然可以通过钩子（hooks）将预处理器和词语切分器当作参数传入，
# 但是为了能够手动进行分词和归一化，通过改写 (overwrite) build_analyzer方法实现
#  The class provides several hooks with which we could customize the preprocessing and tokenization stages.
# The preprocessor and tokenizer can be set in the constructor as parameters.
# We do not want to place the stemmer into any of them,
# because we would then have to do the tokenization and normalization by ourselves.
# Instead,  we overwrite the method build_analyzer as follows:
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer=super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))



# def tfidf(term, doc, docset):
#     tf = float(doc.count(term)) / (sum(short_doc.count(term) for short_doc in docset))
#     idf = math.log(float(len(docset)) / (len([doc for doc in docset if term in doc])))
#     return tf * idf
#
# # test for tfidf
# # a, abb, abc = ['a'], ['a', 'b', 'b'], ['a', 'b', 'c']
# # D = [a, abb, abc]
# # print(tfidf('c', abc, D))
# # 没有进行平滑处理，下面是用Scikit自带的TFIDF
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer=super(TfidfVectorizer,self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
