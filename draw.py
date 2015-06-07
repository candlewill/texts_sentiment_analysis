__author__ = 'NLP-PC'
import pickle

predict_expanding_pos_content = pickle.load(open("./acc_tmp/predict/predict_expanding_pos_content.p", "rb"))
predict_expanding_neg_content = pickle.load(open("./acc_tmp/predict/predict_expanding_neg_content.p", "rb"))
predict_expanded_texts_with_pos = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_pos.p", "rb"))
predict_expanded_texts_with_neg = pickle.load(open("./acc_tmp/predict/predict_expanded_texts_with_neg.p", "rb"))

