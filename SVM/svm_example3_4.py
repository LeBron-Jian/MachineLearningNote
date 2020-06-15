# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D ** 2]

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)


def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1) ** 2)


gamma = 0.8

# 下面进行训练，得到一个支持向量机的模型（这里我们没有训练，直接画出来了）
# 因为测试的数据是我们自己写的，为了方便，我们自己画出来，当然你也可以自己做
xls = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
x2s = gaussian_rbf(xls, -2, gamma)
x3s = gaussian_rbf(xls, 1, gamma)

XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X2D, 1, gamma)]
yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(11, 4))

# plt.subplot(121)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c='red')
plt.plot(X1D[:, 0][yk == 0], np.zeros(4), 'bs')
plt.plot(X1D[:, 0][yk == 1], np.zeros(5), 'g*')
plt.plot(xls, x2s, 'g--')
plt.plot(xls, x3s, 'b:')
plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'Similarity', fontsize=14)

plt.annotate(r'$\mathbf{x}$',
             xy=(X1D[3, 0], 0),
             xytest=(-0.5, 0.20),
             ha='center',
             arrowprops=dict(facecolor='black', shrink=0.1),
             fontsize=18,
             )
plt.text(-2, 0.9, "$x_2$", ha='center', fontsize=20)
plt.text(1, 0.9, "$x_3$", ha='center', fontsize=20)
plt.axis([-4.5, 4.5, -0.1, 1.1])

# plt.subplot(122)
# plt.grid(True, which='both')
# plt.axhline(y=0, color='k')
# plt.axvline(x=0, color='k')
# plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], 'bs')
# plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], 'g*')
# plt.xlabel(r'$x_2$', fontsize=20)
# plt.ylabel(r'$x_3$', fontsize=20, rotation=0)
#
# plt.annotate(r'$\phi\left(\mathbf[x]\right)$',
#              xy=(XK[3, 0], XK[3, 1]),
#              xytest=(0.65, 0.50),
#              ha='center',
#              arrowprops=dict(facecolor='black', shrink=0.1),
#              fontsize=18,
#              )
#
# plt.plot([-0.1, 1.1], [0.57, -0.1], 'r--', linewidth=3)
# plt.axis([-0.1, 1.1, -0.1, 1.1])

# plt.subplots_adjust(right=1)
plt.show()
