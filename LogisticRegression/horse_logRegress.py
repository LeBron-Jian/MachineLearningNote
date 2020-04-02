#-*_coding:utf-8_*_
import math
from numpy import *

# Sigmoid函数的计算
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

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


# Logistic回归分类函数
def  classifyVetor(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest(filetrain,filetest):
    frTrain = open(filetrain)
    frTest = open(filetest)
    trainingSet = []
    trainingLabeles = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabeles.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabeles,500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVetor(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is  %f'%errorRate)
    return errorRate

def multTest(filetrain,filetest):
    numTests = 15
    errorSum = 1.0
    for k in range(numTests):
        errorSum += colicTest(filetrain,filetest)
    print('after %d iterations  the average error rate is %f'%(numTests,errorSum/float(numTests)))


if __name__ == '__main__':
    filetrain = 'horseColicTraining.txt'
    filetest = 'horseColicTest.txt'
    multTest(filetrain,filetest)
