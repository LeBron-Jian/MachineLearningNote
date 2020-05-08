import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# make_nlobs方法常被用来生成聚类算法的测试数据
# make_blobs会根据用户指定的特征数量，中心点数量，范围等来生成几类数据
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类型 共10000个样本，每个样本3个特征，共4个簇
# n_samples表示产生多少个数据  n_features表示数据是几维，centers表示中心点 cluster_std表示分布的标准差
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1],
                [2, 2, 2]], cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
# plt.show()

# pca = PCA(n_components=3)
# pca.fit(X)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)

pca = PCA(n_components = 'mle')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)

# X_new = pca.transform(X)
# a = X_new[:, 0]
# b = X_new[:, 1]
# plt.scatter(a,b, marker='o')
# # plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
# plt.show()