import os ,sys
from numpy import *
import operator

# k-近邻算法
def classify0(inX,dataSet,labels,k):
    # shape读取数据矩阵第一维度的长度
    dataSetSize = dataSet.shape[0]
    # tile重复数组inX，有dataSet行 1个dataSet列，减法计算差值
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # **是幂运算的意思，这里用的欧式距离
    sqDiffMat = diffMat ** 2
    # 普通sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort返回数值从小到大的索引值（数组索引0,1,2,3）
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 根据排序结果的索引值返回靠近的前k个标签
        voteLabel = labels[sortedDistIndicies[i]]
        # 各个标签出现频率
        classCount[voteLabel] = classCount.get(voteLabel,0) +1
    ##!!!!!  classCount.iteritems()修改为classCount.items()
    #sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list。
    # reverse默认升序 key关键字排序itemgetter（1）按照第一维度排序(0,1,2,3)
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    # 每个手写识别为32*32大小的二进制图像矩阵，转换为1*1024 numpy向量数组returenVect
    returnVect = zeros((1,1024))
    fr = open(filename)
    # 循环读出前32行
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            # 将每行的32个字符值存储在numpy数组中
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 测试算法
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    # 定义文件数 x 每个向量的训练集
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 解析文件名
        classNumStr = int(fileStr.split('_')[0])
        # 存储类别
        hwLabels.append(classNumStr)
        # 访问第i个文件内的数据
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
    # 测试数据集
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # 从文件名中分离出数字作为基准
        classNumStr = int(fileStr.split('_')[0])
        # 访问第i个文件内的测试数据，不存储类 直接测试
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is: %d" %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total rate is:%f"% (errorCount/float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()

    # res = img2vector('testDigits/0_13.txt')
    # print(res[0,0:31])
    # print(res[0,32:63])
