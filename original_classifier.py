__author__ = 'NLP-PC'
import numpy as np

tweets,labels=open_csv('training.100000.processed.noemoticon.csv')
tweets,labels=np.array(tweets),np.array(labels)
feature=feature_model(model_type)
