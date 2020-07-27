from sklearn.svm import SVR  # SVM中的回归算法
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# 数据预处理，使得数据更加有效的被模型或者评估器识别
from sklearn import preprocessing
from sklearn.externals import joblib
 
# 获取数据
origin_data = pd.read_csv('wine.txt',header=None)
X = origin_data.iloc[:,1:].values
Y = origin_data.iloc[:,0].values
print(type(Y))
# print(type(Y.values))
# 总特征  按照特征的重要性排序的所有特征
all_feature = [ 9, 12,  6, 11,  0, 10,  5,  3,  1,  8,  4,  7,  2]
# 这里我们选取前三个特征
topN_feature = all_feature[:3]
print(topN_feature)
 
# 获取重要特征的数据
data_X = X[:,topN_feature]
 
# 将每个特征值归一化到一个固定范围
# 原始数据标准化，为了加速收敛
# 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
data_X = preprocessing.MinMaxScaler().fit_transform(data_X)
 
# 利用train_test_split 进行训练集和测试集进行分开
X_train,X_test,y_train,y_test  = train_test_split(data_X,Y,test_size=0.3)
 
# 通过多种模型预测
model_svr1 = SVR(kernel='rbf',C=50,max_iter=10000)
 
 
# 训练
# model_svr1.fit(data_X,Y)
model_svr1.fit(X_train,y_train)
 
# 得分
score = model_svr1.score(X_test,y_test)
print(score)