from sklearn import datasets
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def colicSklearn():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    trainingSet,testSet,trainingLabels,testLabels = train_test_split(X,Y,test_size=0.25,random_state=40)
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print("正确率为%s%%" % test_accurcy)

if __name__  == '__main__':
    colicSklearn()