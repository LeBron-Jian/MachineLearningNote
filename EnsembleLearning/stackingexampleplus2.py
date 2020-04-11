from sklearn import datasets
  
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
  
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
  
basemodel1 = XGBClassifier()
basemodel2 = lgb.LGBMClassifier()
basemodel3 = RandomForestClassifier(random_state=1)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[basemodel1, basemodel2, basemodel3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
 
print('5-fold cross validation:\n')
 
for basemodel, label in zip([basemodel1, basemodel2, basemodel3, sclf],
                      ['xgboost',
                       'lightgbm',
                       'Random Forest',
                       'StackingClassifier']):
  
    scores = model_selection.cross_val_score(basemodel,X, y,
                                              cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))