#_*_coding:utf-8_*_
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
import matplotlib
 
 
# np.meshgrid() 生成网格坐标点
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
 
# generate normal  (not abnormal) training observations 
X = 0.3*np.random.randn(100, 2)
X_train = np.r_[X+2, X-2]
 
# generate new normal (not abnormal) observations
X = 0.3*np.random.randn(20, 2)
X_test = np.r_[X+2, X-2]
 
# generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
 
 
# fit the model for novelty detection  (novelty=True)
clf = LOF(n_neighbors=20, contamination=0.1, novelty=True)
clf.fit(X_train)
 
# do not use predict, decision_function and score_samples on X_train
# as this would give wrong results but only on new unseen data(not
# used in X_train , eg: X_test, X_outliers or the meshgrid)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
'''
### contamination=0.1
X_test: [ 1  1  1  1  1  1  1  1  1  1  1  1  1  1 -1  1  1 -1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1 -1  1  1 -1  1  1]
 
### contamination=0.01
X_test: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1]
 
y_pred_outliers: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
'''
 
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
 
# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
plt.title('Novelty Detection with LOF')
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
 
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
 
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')
 
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
            ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
            loc='upper left',
            prop=matplotlib.font_manager.FontProperties(size=11))
 
plt.xlabel("errors novel regular:%d/40; errors novel abnormal: %d/40"
    %(n_error_test, n_error_outliers))
plt.show()
