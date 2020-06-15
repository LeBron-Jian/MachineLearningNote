from sklearn.svm import SVC
# 生成数据集
from sklearn.datasets.samples_generator import make_blobs, make_circles
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def train_SVM():
    # n_samples=50 表示取50个点，centers=2表示将数据分为两类
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

    # 线性核函数
    model = SVC(kernel='linear')
    model.fit(X, y)

    # plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    # plot_SVC_decision_function(model)
    # plt.show()
    return X, y


def plot_base_image(X, y):
    # 画图形
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    # 线性等分详细
    xfit = np.linspace(-1, 3.5)
    plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                         color='#AAAAAA', alpha=0.4)  # alpha为透明度
    plt.show()


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


def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.6)
    X, y = X[:N], y[:N]
    model = SVC(kernel='linear')
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_SVC_decision_function(model, ax)


def train_svm_plus():
    # 二维圆形数据 factor 内外圆比例（0， 1）
    X, y = make_circles(100, factor=0.1, noise=0.1)
    # 加入径向基函数
    clf = SVC(kernel='rbf')
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_SVC_decision_function(clf, plot_support=False)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    return X, y


def plot_3D(X, y, elev=30, azim=30):
    # 我们加入了新的维度 r
    r = np.exp(-(X ** 2).sum(1))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


if __name__ == '__main__':
    # train_SVM()

    # fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    # for axi, N in zip(ax, [60, 120]):
    #     plot_svm(N, axi)
    #     axi.set_title('N = {0}'.format(N))

    X, y = train_svm_plus()
    # plot_3D(elev=30, azim=30, X=X, y=y)
