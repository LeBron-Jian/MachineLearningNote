from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
# 读取数据
X = []
Y = []
fr = open("datingTestSet.txt", encoding='utf-8')
print(fr)
index = 0
for line in fr.readlines():
    # print(line)
    line = line.strip()
    line = line.split('\t')
    X.append(line[:3])
    Y.append(line[-1])
 
# 归一化
scaler = MinMaxScaler()
# print(X)
X = scaler.fit_transform(X)
# print(X)
 
# 交叉分类
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.2)
 
#高斯贝叶斯模型
model = GaussianNB()
model.fit(train_X, train_y)
 
# 预测测试集数据
predicted = model.predict(test_X)
# 输出分类信息
res = metrics.classification_report(test_y, predicted)
# print(res)
# 去重复，得到标签类别
label = list(set(Y))
# print(label)
# 输出混淆矩阵信息
matrix_info = metrics.confusion_matrix(test_y, predicted, labels=label)
# print(matrix_info)