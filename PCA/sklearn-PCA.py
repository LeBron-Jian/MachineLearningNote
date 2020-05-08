from sklearn import decomposition
import matplotlib.pyplot as plt
from numpy import *

def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr =[list(map(float,line)) for line in stringArr]
    return mat(datArr)

def replaceNanWithMean():
    datMat = loadDataSet('testSet1.txt')
    numFeat = shape(datMat)[1]  # 取得行数，对每行进行处理
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat


dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
# print eigVals
print(sum(eigVals) * 0.9)
print(sum(eigVals[:6]))
plt.plot(eigVals[:20])  # 对前20个画图观察
plt.show()

pca_sklearn = decomposition.PCA()
pca_sklearn.fit(replaceNanWithMean())
main_var = pca_sklearn.explained_variance_

print(sum(main_var)*0.9)
print(sum(main_var[:6]))
plt.plot(main_var[:20])
plt.show()