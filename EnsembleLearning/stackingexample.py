from mlxtend.classifier import StackingClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
 
X, y = load_iris(return_X_y=True)
estimators = [
    RandomForestClassifier(n_estimators=10, random_state=42),
    make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42))]
clf = StackingClassifier(
    classifiers=estimators,
    meta_classifier=LogisticRegression()
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = clf.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)  # 0.868421052631579