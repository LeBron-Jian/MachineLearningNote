import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state=9)

for index, k in enumerate((2, 3, 4, 5)):
    plt.subplot(2, 2, index + 1)
    clf = MiniBatchKMeans(n_clusters=k, batch_size=200, random_state=9)
    y_pred = clf.fit_predict(X)
    score = metrics.calinski_harabaz_score(X, y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k, score)),
             transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
plt.show()
