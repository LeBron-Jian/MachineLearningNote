#_*_coding:utf-8_*_
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
for name,label in [('Setosa', 0), ('Versico',1), ('Viriginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment = 'center',
              bbox = dict(alpha=0.5, edgecolor ='w',facecolor='w'))
y = np.choose(y, [1, 2, 0]).astype(np.float)
y1 = plt.scatter(X[:,0], X[:, 1], X[:, 2], c=z, cmap=plt.cm.nipy_spectral,
               edgecolor='r')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()