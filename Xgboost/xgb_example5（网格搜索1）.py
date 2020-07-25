from sklearn.model_selection import GridSearchCV
tuned_parameters= [{'n_estimators':[100,200,500],
                  'max_depth':[3,5,7], ##range(3,10,2)
                  'learning_rate':[0.5, 1.0],
                  'subsample':[0.75,0.8,0.85,0.9]
                  }]
tuned_parameters= [{'n_estimators':[100,200,500,1000]
                  }]
clf = GridSearchCV(XGBClassifier(silent=0,nthread=4,learning_rate= 0.5,min_child_weight=1, max_depth=3,gamma=0,subsample=1,colsample_bytree=1,reg_lambda=1,seed=1000), param_grid=tuned_parameters,scoring='roc_auc',n_jobs=4,iid=False,cv=5) 
clf.fit(X_train, y_train)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)
y_true, y_pred = y_test, clf.predict(X_test)
print"Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred)
y_proba=clf.predict_proba(X_test)[:,1]
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_true, y_proba)          