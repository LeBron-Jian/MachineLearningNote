from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
 
X, y = load_iris().data, load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
 
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
 
# 绝对少数服从多数原则投票 如果某标记的投票过半数，则预计为该标记；否则拒绝预测
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                         voting='hard')
model1 = eclf1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)
print(score1)  # 0.9210526315789473
 
# 相对少数服从多数原则投票 预测为得票最多的标记，若同时出现多个票数最多，则任选其一
eclf2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                         voting='soft')
model2 = eclf2.fit(X_train, y_train)
score2 = model2.score(X_test, y_test)
print(score2)  # 0.9210526315789473
 
# 使用加权投票法
eclf3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                         voting='soft', weights=[2, 1, 1], flatten_transform=True)
model3 = eclf3.fit(X_train, y_train)
score3 = model3.score(X_test, y_test)
print(score3)  # 0.9210526315789473