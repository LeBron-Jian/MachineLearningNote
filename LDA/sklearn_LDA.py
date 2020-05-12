# _*_coding:utf-8_*_
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(X_train_std,y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_std,y_train)

plot_decision_region(X_train_std,y_train,classmethod)