from sklearn.datasets import load_iris
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
  
iris = load_iris()
X = iris.data
y = iris.target
#基分类器1：xgboost
pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)),
                      XGBClassifier())
#基分类器2：RandomForest
pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),
                      RandomForestClassifier())
  
sclf = StackingClassifier(classifiers=[pipe1, pipe2],
                          meta_classifier=LogisticRegression())
  
sclf.fit(X, y)