from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
 
clf = DecisionTreeClassifier()
clfb = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                         max_samples=0.5, max_features=0.5)
 
clfb1 = BaggingClassifier(base_estimator=SVC(),
                          max_samples=0.5, max_features=0.5)
model1 = clf.fit(X, y)
model2 = clfb.fit(X, y)
model3 = clfb1.fit(X, y)
predict = model1.score(X_test, y_test)
predictb = model2.score(X_test, y_test)
predictb1 = model3.score(X_test, y_test)
 
print(predict)  # 1.0
print(predictb)  # 0.9473684210526315
print(predictb1)  # 0.9210526315789473