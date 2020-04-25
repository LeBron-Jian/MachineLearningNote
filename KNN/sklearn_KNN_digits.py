#_*_coding:utf-8_*_

import numpy as np
import operator
import os
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):
    # 创建1*1024零向量
    returnVect = np.zeros((1,1024))
    # 打开文件
    fr = open(filename)
    # 按照行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect

# 手写数字分类测试
def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigts目录下的文件名
    trainingFileList = os.listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = np.zeros((m,1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获取文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构建KNN分类器
    neigh = KNN(n_neighbors=3,algorithm='auto')
    # 拟合模型，trainingMat为训练矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat,hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = os.listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获取文件的名字
        fileNameStr = testFileList[i]
        # 获取分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()