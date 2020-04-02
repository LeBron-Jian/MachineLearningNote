#_*_ codingLutf-8_*_

from sklearn.linear_model import LogisticRegression

def colicSklearn(filetrain,filetest):
    frTrain = open(filetrain)
    frTest = open(filetest)
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='sag',max_iter=5000).fit(trainingSet,trainingLabels)
    test_accurcy = classifier.score(testSet,testLabels)*100
    print("正确率为%s%%"%test_accurcy)

if __name__ == '__main__':
    filetrain = 'horseColicTraining.txt'
    filetest = 'horseColicTest.txt'
    colicSklearn(filetrain,filetest)