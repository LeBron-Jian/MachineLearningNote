import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics


# sklearn 中的make_blobs方法常被用来生成聚类算法的测试数据
# X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，共4个簇
# 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# n_samples表示产生多少个数据 n_features表示数据是几维的
# centers表示数据点中心，可以输入整数，代表有几个中心，也可以输入几个坐标
# cluster_std表示分布的标准差
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state=9)
clf = KMeans(n_clusters=4, random_state=9)
y_pred = clf.fit_predict(X)
res = metrics.calinski_harabaz_score(X, y_pred)
print(res)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.show()

