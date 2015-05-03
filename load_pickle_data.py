__author__ = 'NLP-PC'
import pickle
from scipy.sparse import dia_matrix as dm

label = pickle.load(open("debug/trian_vec.p", "rb"))
for i in range(0,10):
    a=label[i,:]
    print(a,type(a))
