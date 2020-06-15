from sklearn.svm import SVC
# 生成数据集
from sklearn.datasets.samples_generator import make_circles, make_blobs
from matplotlib import pyplot as plt
import numpy as np


def plot_SVC_decision_function(model, ax=None, plot_support=True):
    '''Plot the decision function for a 2D SVC'''
    if ax is None:
        ax = plt.gca()  # get子图
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    # 生成网格点和坐标矩阵
    Y, X = np.meshgrid(y, x)
    # 堆叠数组
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])  # 生成等高线 --

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# # n_samples=50 表示取50个点，centers=2表示将数据分为两类
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
#
# fig, ax = plt.subplots(1, 3, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
#
# for axi, C in zip(ax, [100.0, 10.0, 0.1]):
#     model = SVC(kernel='linear', C=C)
#     model.fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     plot_SVC_decision_function(model, axi)
#     axi.scatter(model.support_vectors_[:, 0],
#                 model.support_vectors_[:, 1],
#                 s=300, lw=1, facecolors='none')
#     axi.set_title('C={0:.1f}'.format(C), size=14)

# n_samples=50 表示取50个点，centers=2表示将数据分为两类
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.1)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, gamma in zip(ax, [10.0, 1.0, 0.1]):
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_SVC_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    axi.set_title('gamma={0:.1f}'.format(gamma), size=14)
