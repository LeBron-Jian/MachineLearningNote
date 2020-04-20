import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from numpy import genfromtxt
 
def read_dataset(filePath, delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)
 
# use the same dataset
tr_data = read_dataset('tr_data.csv')
 
clf = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
'''
OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, nu=0.05, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
'''
clf.fit(tr_data)
pred = clf.predict(tr_data)
 
# inliers are labeled 1 , outliers are labeled -1
normal = tr_data[pred == 1]
abnormal = tr_data[pred == -1]
 
plt.plot(normal[:, 0], normal[:, 1], 'bx)
plt.plot(abnormal[:, 0], abnormal[:, 1], 'ro')