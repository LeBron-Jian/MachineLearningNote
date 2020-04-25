# 引入数据集，sklearn包含众多数据集
from sklearn import datasets
# 将数据分为测试集和训练集
from sklearn.model_selection import train_test_split
# 利用邻近点方式训练数据
from sklearn.neighbors import KNeighborsClassifier
  
# 引入数据,本次导入鸢尾花数据，iris数据包含4个特征变量
iris = datasets.load_iris()
# 特征变量
iris_X = iris.data
# print(iris_X)
print('特征变量的长度',len(iris_X))
# 目标值
iris_y = iris.target
print('鸢尾花的目标值',iris_y)
# 利用train_test_split进行训练集和测试机进行分开，test_size占30%
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)
# 我们看到训练数据的特征值分为3类
# print(y_train)
'''
[1 1 0 2 0 0 0 2 2 2 1 0 2 0 2 1 0 1 0 2 0 1 0 0 2 1 2 0 0 1 0 0 1 0 0 0 0
 2 2 2 1 1 1 2 0 2 0 1 1 1 1 2 2 1 2 2 2 0 2 2 2 0 1 0 1 0 0 1 2 2 2 1 1 1
 2 0 0 1 0 2 1 2 0 1 2 2 2 1 2 1 0 0 1 0 0 1 1 1 0 2 1 1 0 2 2]
 '''
# 训练数据
# 引入训练方法
knn = KNeighborsClassifier()
# 进行填充测试数据进行训练
knn.fit(X_train,y_train)
  
params = knn.get_params()
print(params)
'''
{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski',
 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5,
 'p': 2, 'weights': 'uniform'}
  
'''
  
score = knn.score(X_test,y_test)
print("预测得分为：%s"%score)
'''
预测得分为：0.9555555555555556
[1 2 1 1 2 2 1 0 0 0 0 1 2 0 1 0 2 0 0 0 2 2 0 2 2 2 2 1 2 2 2 1 2 2 1 2 0
 2 1 2 1 1 0 2 1]
[1 2 1 1 2 2 1 0 0 0 0 1 2 0 1 0 2 0 0 0 1 2 0 2 2 2 2 1 1 2 2 1 2 2 1 2 0
 2 1 2 1 1 0 2 1]
'''
  
# 预测数据，预测特征值
print(knn.predict(X_test))
'''
[0 2 2 2 2 0 0 0 0 2 2 0 2 0 2 1 2 0 2 1 0 2 1 0 1 2 2 0 2 1 0 2 1 1 2 0 2
 1 2 0 2 1 0 1 2]
'''
# 打印真实特征值
print(y_test)
'''
[1 2 2 2 2 1 1 1 1 2 1 1 1 1 2 1 1 0 2 1 1 1 0 2 0 2 0 0 2 0 2 0 2 0 2 2 0
 2 2 0 1 0 2 0 0]
  
'''