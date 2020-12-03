import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
 
X = [[-1.1], [0.2], [100.1], [0.3]]
clf = LOF(n_neighbors=2)
res = clf.fit_predict(X)
print(res)
print(clf.negative_outlier_factor_)
 
'''
如果 X = [[-1.1], [0.2], [100.1], [0.3]]
[ 1  1 -1  1]
[ -0.98214286  -1.03703704 -72.64219576  -0.98214286]
 
如果 X = [[-1.1], [0.2], [0.1], [0.3]]
[-1  1  1  1]
[-7.29166666 -1.33333333 -0.875      -0.875     ]
 
如果 X = [[0.15], [0.2], [0.1], [0.3]]
[ 1  1  1 -1]
[-1.33333333 -0.875      -0.875      -1.45833333]
'''
