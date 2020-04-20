#_*_coding:utf-8_*_
# 从sklearn.datasets中导入新闻数据抓取器
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text里导入文本特征向量化模板
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
 
# 数据获取
news = fetch_20newsgroups(subset='all')
# 输出数据的条数  18846
print(len(news.data))
 
# 数据预处理：训练集和测试集分割，文本特征向量化
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33
)
# # 查看训练样本
# print(X_train[0])
# # 查看标签
# print(y_train[0:100])
 
# 文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
 
# 使用朴素贝叶斯进行训练
# 使用默认配置初始化朴素贝叶斯
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对参数进行预测
y_predict = mnb.predict(X_test)
 
# 获取结果报告
score = mnb.score(X_test, y_test)
print('The accuracy of Naive bayes Classifier is %s' %score)
 
res = classification_report(y_test, y_predict, target_names=news.target_names)
print(res)