#_*_coding:utf-8_*_
from numpy import *

# 读取数据
def loadDataSet(filename):
    '''
        对于testSet.txt，每行前两个值分别是X1和X2，第三个值数据对应的类别标签
        而且为了设置方便，该函数还将X0的值设置为1.0
        :return:
        '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# Sigmoid函数的计算
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升法
def gradAscent(dataMatIn,classLabels):
    '''
        :param dataMatIn: 是一个2维Numpy数组，每列分别代表每个不同的特征
        每行则代表每个训练样本。
        :param classLabels: 是类别标签，是一个1*100的行向量，为了便于矩阵运算，需要将行向量
        转换为列向量，就是矩阵的转置，再将其赋值与labelMat。
        :return:
        '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # alpha是向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

# 随机梯度上升算法 Stochastic gradient ascent algorithm SGA
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    # weights = wei.getA()
    weights = wei
    dataMat,labelMat = loadDataSet(filename)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) ==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__  == '__main__':
    filename = 'testSet.txt'
    dataArr,labelMat = loadDataSet(filename)
    weights_res = stocGradAscent1(array(dataArr),labelMat,500)
    print(weights_res)
    plotBestFit(weights_res)

'''
[[ 4.12414349]
 [ 0.48007329]
 [-0.6168482 ]]
 '''