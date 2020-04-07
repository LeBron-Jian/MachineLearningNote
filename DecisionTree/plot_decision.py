import matplotlib.pyplot as plt

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

def myTree():
    # treeData = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    treeData = {'voice': {'Think': {'hair': {'Long': 'female', 'Short': 'male'}}, 'Thin': 'female'}}
    return treeData

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
    myData  = myTree()
    myData['voice'][3] = 'maybe'
    print(type(myData))
    # LeafNum = getNumLeafs(myData)
    # TreeDepth = getTreeDepth(myData)
    # print(LeafNum)
    # print(TreeDepth)
    createPlot(myData)