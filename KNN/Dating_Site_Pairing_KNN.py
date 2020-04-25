#_*_coding:utf-8_*_

from numpy import *
import operator

# 获取数据
# def creataDataSet():
#     group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#     labels = ['A','A','B','B']
#     return group,labels
def creataDataSet():
    '''
        labels中   A代表爱情片  B代表动作片
        :return:
        '''
    group = array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['A','A','B','B']
    return group,labels

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


# 将文本记录到转换Numpy的解析程序
def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # print(numberOfLines)  #1000
    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        # 删除空白行
        line = line.strip()
        listFromLine = line.split('\t')
        # 选取前3个元素（特征）存储在返回矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # -1索引表示最后一列元素,位label信息存储在classLabelVector
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector



# 归一化特征值
# 归一化公式 ： （当前值 - 最小值） / range
def autoNorm(dataSet):
    # 存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    minVals = dataSet.min(0)
    # 存放每列最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # 初始化归一化矩阵为读取的dataSet
    normDataSet = zeros(shape(dataSet))
    # 保留第一行
    m = dataSet.shape[0]
    # 特征值相除，特征矩阵是3*1000 min  max range是1*3
    # 因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet = dataSet - tile(minVals , (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2Matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        print('the classifier came back with:%d,the read answer is :%d '
              %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount +=1.0
    print("the total error rate is :%f"%(errorCount/float(numTestVecs)))


# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small','in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters od ice cream consumed per year?"))
    dataingDataMat,datingLabels = file2Matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person ",resultList[classifierResult-1])

# 使用sklearn中的preprocessing库的Normalizer类对数据进行归一化的代码：
from sklearn.preprocessing import Normalizer
# 归一化 返回值为归一化后的数据
# normalize = Normalizer().fit_transform(iris,data)


if __name__ == '__main__':
    # datingClassTest()
    # filename = 'datingTestSet2.txt'
    # datingDataMat,datingLabels = file2Matrix(filename)
    # classifyPerson()

    # print(datingDataMat)
    # print(datingLabels)
    # normMat,ranges,minVals = autoNorm(datingDataMat)
    # print(normMat)
    # print(ranges)
    # print(minVals)


    # 使用Matplotlib创建散点图

    # import matplotlib.pyplot as plt
    # from pylab import mpl
    #
    # # 指定默认字体
    # mpl.rcParams['font.sans-serif'] = ['FangSong']
    # # 解决保存图像是负号- 显示为方块的问题
    # mpl.rcParams['axes.unicode_minus'] = False
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.xlabel("玩视频游戏所耗时间百分比")
    # plt.ylabel("每周消费的冰淇淋公升数")
    # plt.show()

    group,labels = creataDataSet()
    print(group)
    print(labels)
    # knn = classify0([0,0],group,labels,3)
    # print(knn)

