import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random
import time
from sklearn import preprocessing as pp

path = 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(pdData.head(10))

positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]


# fig, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 如果想查看Sigmoid函数，可以运行下面代码
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(nums, sigmoid(nums), 'r')

def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 增加一行1  在第0列
pdData.insert(0, 'Ones', 1)
# print(pdData.head())
# 设置X和Y  即训练数据和训练变量
orig_data = pdData.values
cols = orig_data.shape[1]  # (100, 4)
X = orig_data[:, 0:cols - 1]  # 等价于 np.matrix(X.values)
y = orig_data[:, cols - 1:cols]  # 等价于 np.matrox(data.iloc[:, 3:4].value

# 传递numpu矩阵 并初始化参数矩阵 theta
theta = np.zeros([1, 3])  # [[0. 0. 0.]]


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 比较三种不同梯度下降方法
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


# 洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break
    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = 'Original' if (data[:, 1] > 2).sum() > 1 else 'Scaled'
    name += 'data - learning rate: {} -'.format(alpha)
    if batchSize == n:
        strDescType = 'Gradient'
    elif batchSize == 1:
        strDescType = 'Stochastic'
    else:
        strDescType = 'Min-batch({})'.format(batchSize)
    name += strDescType + 'descent - Stop: '
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = 'gradient norm < {}'.format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter:{}-Last cost: {:03.2f}--Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur
    ))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + '- Error vs. Iteration')
    return theta


# 设置阈值
def predict(x, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


if __name__ == '__main__':
    # 设定迭代次数   选择的梯度下降方法是基于所有样本的
    n = 100
    runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
    # 根据损失值停止    设定阈值 1e-6 差不多需要110 000 次迭代
    # runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
    # 根据梯度变换停止  设定阈值 0.05，差不多需要 40000次迭代
    # runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)
    # 对比不同的梯度下降方法  stochastic descent  这个不收敛
    # runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
    # 我们将学习率调小，提高thresh再试试，发现收敛了
    # runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000001)
    # Mine-batch descent  这个也不收敛 ，我们将alpha调小 0.000001  发现收敛了
    # runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
    scaled_data = orig_data.copy()
    # scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
    # runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.01)
    # runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.01)
    # runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.01)
    # runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)
    scaled_X = scaled_data[:, :3]
    y = scaled_data[:, 3]
    predictions = predict(scaled_X, theta)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for a, b in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))
    print(len(correct))
