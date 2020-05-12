import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification
 
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0,
                           n_classes=3, n_informative=2, n_clusters_per_class=1,
                           class_sep=0.5, random_state=10)
 
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)

# pca没有使用类别信息，对此数据降维后，样本特征和类别的信息关联几乎丢失
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
# [0.43377069 0.3716351 ]
# [1.21083449 1.0373882 ]
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()

# LDA是有监督学习，将为后样本特征信息之间的关系得以保留
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()