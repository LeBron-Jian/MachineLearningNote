# _*_coding:utf-8_*_
import operator
from math import log
import  matplotlib.pyplot as  plt

# 计算香农熵  度量数据集无序程度
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for fearVec in dataSet:
        currentLabel = fearVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

# 划分数据集
def  splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducesFeatVec = featVec[:axis]
            reducesFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducesFeatVec)
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

# 获取叶子节点的数目
def getNumLeafs(myTree):
    # 初始化结点数
    numLeafs = 0
    firstSides = list(myTree.keys())
    # 找到输入的第一个元素，第一个关键词为划分数据集类别的标签
    firstStr = firstSides[0]
    secondDect = myTree[firstStr]
    for key in secondDect.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDect[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDect[key])
        else:
            numLeafs +=1
    return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth

    return maxDepth

# 定义文本框和箭头格式（树节点格式的常量）
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrows_args = dict(arrowstyle='<-')

# 绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,
                            arrowprops=arrows_args)


# 在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString, va="center", ha="center", rotation=30)


def plotTree(myTree,parentPt,nodeTxt):
    # 求出宽和高
    numLeafs = getNumLeafs(myData)
    depth = getTreeDepth(myData)
    firstStides = list(myTree.keys())
    firstStr = firstStides[0]
    # 按照叶子结点个数划分x轴
    cntrPt = (plotTree.xOff + (0.1 + float(numLeafs)) /2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    # y方向上的摆放位置 自上而下绘制，因此递减y值
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__  == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW  # x方向计算结点坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  # 绘制
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))  # 添加文本信息
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD  # 下次重新调用时恢复y

# 主函数
def createPlot(inTree):
    # 创建一个新图形并清空绘图区
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t')  for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    print(lenses)
    myData = createTree(lenses, lensesLabels)
    print(myData)
    createPlot(myData)