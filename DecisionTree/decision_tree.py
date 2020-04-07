from math import log
import operator

# 计算数据的熵（entropy）
def calcShannonEnt(dataSet):
    # 数据条数，计算数据集中实例的总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 每行数据的最后一个类别（也就是标签）
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        # 统计有多少个类以及每个类的数量
        labelCounts[currentLable]  += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算单个类的熵值
        prob = float(labelCounts[key]) / numEntries
        # 累加每个类的熵值
        shannonEnt -= prob * log(prob , 2)
    return shannonEnt

# 创建数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    # dataSet = [['Long', 'Think', 'male'],
    #            ['Short', 'Think', 'male'],
    #            ['Short', 'Think', 'male'],
    #            ['Long', 'Thin', 'female'],
    #            ['Short', 'Thin', 'female'],
    #            ['Short', 'Think', 'female'],
    #            ['Long', 'Think', 'female'],
    #            ['Long', 'Think', 'female']]
    # labels = ['hair', 'voice']
    return dataSet,labels

# 按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    '''

    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        # 如果发现符合要求的特征，将其添加到新创建的列表中
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)

    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureTpSplit(dataSet):
    '''
        此函数中调用的数据满足以下要求
        1，数据必须是一种由列表元素组成的列表，而且所有列表元素都要具有相同的数据长度
        2，数据的最后一列或者实例的最后一个元素是当前实例的类别标签
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1
    # 原始的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 计算每种划分方式的信息熵，并对所有唯一特征值得到的熵求和
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            # 按照特征分类后的熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 计算最好的信息增益,信息增益越大，区分样本的能力越强，根据代表性
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 按照分类后类别数量排序
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 创建树的函数代码
def createTree(dataSet,labels):
    '''

        :param dataSet:  输入的数据集
        :param labels:  标签列表（包含了数据集中所有特征的标签）
        :return:
        '''
    # classList 包含了数据集中所有类的标签
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0])  == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureTpSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 字典myTree存储了树的所有信息，这对于后面绘制树形图很重要
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels =labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),
                                                  subLabels)

    return myTree

# 使用决策树的分类函数
def classify(inputTree ,featLabels,testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    # print('featLables:',featLabels)
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        # 若该特征值等于当前key，yes往下走
        if testVec[featIndex]  == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#使用决策树来分类
def classify11(inputTree,featLabels,testVec):
    #python3.X
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    # python3.X
    secondDict=inputTree[firstStr]  #baocun在secondDict中
    featIndex=featLabels.index(firstStr)  #建立索引
    for key in secondDict.keys():
        if testVec[featIndex]==key: #若该特征值等于当前key，yes往下走
            if type(secondDict[key]).__name__=='dict':# 若为树结构
                classLabel=classify(secondDict[key],featLabels,testVec) #递归调用
            else:  classLabel=secondDict[key]#为叶子结点，赋予label值
    return classLabel #分类结果

def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1: 'yes'}},1:'no'}}}}
    ]
    return listOfTrees[i]

# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw =open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    myData,labels = createDataSet()
    # print(myData)
    # print(labels)
    myTree = createTree(myData,labels)
    print(myTree)
    print(type(myTree))
    myTree1 = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    storeTree(myTree1,'classifierStorage.txt')
    res = grabTree('classifierStorage.txt')
    print(res)
    # myTree = retrieveTree(0)
    # myData, labels = createDataSet()
    # print(myData)
    # print(labels)
    # res1 = classify(myTree,labels,[1,1])
    # print(res1)
    # res2 = classify(myTree,labels,[1,0])
    # print(res2)


'''
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
['no surfacing', 'flippers']
yes
no
'''