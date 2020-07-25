import xgboost as xgb
import pandas as pd
#获取数据
from sklearn import cross_validation
from sklearn.datasets import load_iris
iris = load_iris()
#切分数据集
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
#设置参数
m_class = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27)
#训练
m_class.fit(X_train, y_train)
test_21 = m_class.predict(X_test)
print "Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21)
#预测概率
#test_2 = m_class.predict_proba(X_test)
#查看AUC评价标准
from sklearn import metrics
print "Accuracy : %.2f" % metrics.accuracy_score(y_test, test_21)
##必须二分类才能计算
##print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)
#查看重要程度
feat_imp = pd.Series(m_class.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
import matplotlib.pyplot as plt
plt.show()
#回归
#m_regress = xgb.XGBRegressor(n_estimators=1000,seed=0)
#m_regress.fit(X_train, y_train)
#test_1 = m_regress.predict(X_test)