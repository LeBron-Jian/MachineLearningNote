import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
url1 = pd.read_csv(r'wine.txt', header=None)
# url1 = pd.DataFrame(url1)
# df = pd.read_csv(url1,header=None)
url1.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# 重要性： [0.10658906 0.02539968 0.01391619 0.03203319 0.02207807 0.0607176
#  0.15094795 0.01464516 0.02235112 0.18248262 0.07824279 0.1319868
#  0.15860977]
# print(url1)

# 查看几个标签
# Class_label = np.unique(url1['Class label'])
# print(Class_label)
# 查看数据信息
# info_url = url1.info()
# print(info_url)

# 除去标签之外，共有13个特征，数据集的大小为178，
# 下面将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

print(type(url1))
# url1 = url1.values
# x = url1[:,0]
# y = url1[:,1:]
x, y = url1.iloc[:, 1:].values, url1.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
feat_labels = url1.columns[1:]
# n_estimators：森林中树的数量
# n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)

# 下面对训练好的随机森林，完成重要性评估
# feature_importances_  可以调取关于特征重要程度
importances = forest.feature_importances_
print("重要性：", importances)
x_columns = url1.columns[1:]
indices = np.argsort(importances)[::-1]
x_columns_indices = []
for f in range(x_train.shape[1]):
    # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
    # 到根，根部重要程度高于叶子。
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    x_columns_indices.append(feat_labels[indices[f]])

print(x_columns_indices)
print(x_columns.shape[0])
print(x_columns)
print(np.arange(x_columns.shape[0]))

# 筛选变量（选择重要性比较高的变量）
threshold = 0.15
x_selected = x_train[:, importances > threshold]

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title("红酒的数据集中各个特征的重要程度", fontsize=18)
plt.ylabel("import level", fontsize=15, rotation=90)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
for i in range(x_columns.shape[0]):
    plt.bar(i, importances[indices[i]], color='orange', align='center')
    plt.xticks(np.arange(x_columns.shape[0]), x_columns_indices, rotation=90, fontsize=15)
plt.show()
