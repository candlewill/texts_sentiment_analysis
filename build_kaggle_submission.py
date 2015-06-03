__author__ = 'NLP-PC'
import pickle
import csv

label = pickle.load(open("acc_tmp/kaggle_predict_label.p", "rb"))
id=pickle.load(open("acc_tmp/review_id.p", "rb"))
result=[]
for i in range(0,len(label)):
    polarity=label[i]
    id_s=id[i]
    result.append([id_s,polarity])

with open('acc_tmp/submission_version_1.2.csv', 'w', newline='') as f:
    writer = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows(result)

print("OK")
