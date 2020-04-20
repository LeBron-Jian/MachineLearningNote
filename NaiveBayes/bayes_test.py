import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[-1, -1], [-2, -2], [-3, -3], [-4, -4], [1, 1], [2, 2], [3, 3]])
y = np.array([1, 1, 1, 1, 2, 2, 2])
clf = GaussianNB()
re = clf.fit(X, y)
# print(re)
# GaussianNB(priors=None, var_smoothing=1e-09)

re1 = clf.priors
# print(re1)  #None

# 设置priors参数值
re2 = clf.set_params(priors=[0.625, 0.375])
# print(re2)
# GaussianNB(priors=[0.625, 0.375], var_smoothing=1e-09)

# 返回各类标记对应先验概率组成的列表
re3 = clf.priors
# print(re3)
# [0.625, 0.375]

re4 = clf.class_prior_
# print(re4)
# [0.57142857 0.42857143]

re5 = type(clf.class_prior_)
# print(re5)
# <class 'numpy.ndarray'>

re6 = clf.class_count_
# print(re6)
# [4. 3.]

re7 = clf.theta_
# print(re7)
# [[-2.5 -2.5]
#  [ 2.   2. ]]

re8 = clf.sigma_
# print(re8)
# [[1.25000001 1.25000001]
#  [0.66666667 0.66666667]]

re9 = clf.get_params(deep=True)
# print(re9)
# {'priors': [0.625, 0.375], 'var_smoothing': 1e-09}

re10 = clf.get_params()
# print(re10)
# {'priors': [0.625, 0.375], 'var_smoothing': 1e-09}

re11 = clf.set_params(priors=[0.625, 0.375])
# print(re11)
# GaussianNB(priors=[0.625, 0.375], var_smoothing=1e-09)

re12 = clf.fit(X, y, np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
re13 = clf.theta_
re14 = clf.sigma_
print(re12)
print(re13)
print(re14)