from numpy import *
import matplotlib.pyplot as plt
import pandas as pd


def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    # for line in fr.readlines():
    #     a =line.strip().split(delim)
    #     b = a
    #     print(len(a))
    #     print(type(b))
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print(stringArr)
    # print(type(stringArr))

    datArr = [list(line) for line in stringArr]
    # print(type(datArr))
    datArr = [list(map(float, line)) for line in stringArr]
    # print( mat(datArr).shape)
    return mat(datArr)


def pca(dataMat,topNfeat=999999):
    meanVals = mean(dataMat, axis=0)
    # meanVals = mean(dataMat)
    # 减去平均值
    meanRemoved = dataMat - meanVals
    # print(meanRemoved.shape)
    # 求协方差
    covMat = cov(meanRemoved,rowvar = 0)
    # 计算特征值和特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    # print(eigVals.shape)
    # print(eigVects.shape)
    # 排序 排序从最小到最大
    eigValInd = argsort(eigVals)
    # 保留最大的前K个特征值
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 对应的特征向量
    redEigVects = eigVects[:,eigValInd]
    # 将数据转换到低维度新空间
    lowDDataMat = meanRemoved * redEigVects
    # 重构数据，用于调试
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat

dataMat = loadDataSet('testSet.txt')
lowDMat,reconMat = pca(dataMat,1)
# # print('shape(lowDMat):',shape(lowDMat))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
# plt.show()
