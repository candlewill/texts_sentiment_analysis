import re
import numpy as np
import math

dir = './data/'
file_name = 'LLNL-Thunder-2007-1.1-cln.txt'
# 去掉了首行的空白
reg = "[' ']+"
start_time, execute_time, n_processors = [], [], []
with open(dir + file_name, 'rt', newline='\n') as f:
    for line in f:
        s, e, n = np.array(re.split(reg, line.strip()), dtype=np.float32)[[1, 3, 7]]
        # 数据清理，去除-1和n_processors==0的值
        if s != -1 and e != -1 and n != -1 and n != 0:
            start_time.append(s), execute_time.append(e), n_processors.append(n)
start_time, execute_time, n_processors = np.array(start_time, dtype=np.float32), np.array(execute_time, dtype=np.float32), np.array(n_processors, dtype=np.float32)
earliest = min(start_time)
latest = max(start_time + execute_time)
print(earliest, latest)

interval = 300

num_section = math.ceil((latest - earliest)/interval)
print(num_section)

hist = np.zeros(num_section, dtype=np.int)
num_row = len(start_time)

for i in range(num_row):
    most_left_section = math.floor(start_time[i]/interval)
    most_right_section = math.floor((start_time[i]+execute_time[i])/interval)
    for j in range(most_left_section, most_right_section+1):
        hist[j]=hist[j]+n_processors[i]
print(hist)

import csv
with open('acc_tmp/pengjie.csv', 'w', newline='') as f:
    writer = csv.writer(f,quoting=csv.QUOTE_MINIMAL)
    for k in range(num_section):
        out= (k, interval*k, hist[k])
        writer.writerow(out)

print("OK")
# 画图
from matplotlib import pyplot as plt
plt.plot(range(num_section), hist, alpha= 0.6)
plt.xlabel('time')
plt.ylabel('num of processors')
plt.show()