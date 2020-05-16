from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

# 生成新的数据
X, y_true = make_blobs(n_samples=800, centers=4, random_state=11)
# plt.scatter(X[:, 0], X[:, 1])

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

X = X_stretched
# plt.scatter(X[:, 0], X[:, 1])

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# print(centers)

gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
