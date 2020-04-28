import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans, DBSCAN

# 下面我们生成三组数据
X1, y1 = make_circles(n_samples=5000, factor=0.6, noise=0.05)
X2, y2 = make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]],
                    cluster_std=[[0.1]], random_state=9)
X = np.concatenate((X1, X2))
clf = KMeans(n_clusters=3, random_state=9)
clf1 = DBSCAN(eps=0.1, min_samples=10)
y_pred = clf1.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.show()
