from sklearn.model_selection import GridSearchCV
parameters= [{'learning_rate':[0.01,0.1,0.3],'n_estimators':[1000,1200,1500,2000,2500]}]
clf = GridSearchCV(XGBClassifier(
             max_depth=3,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.6,
             colsample_bytree=0.6,
             objective= 'binary:logistic', #逻辑回归损失函数
             scale_pos_weight=1,
             reg_alpha=0,
             reg_lambda=1,
             seed=27
            ),
            param_grid=parameters,scoring='roc_auc') 
clf.fit(X_train, y_train)
print(clf.best_params_) 
y_pre= clf.predict(X_test)
y_pro= clf.predict_proba(X_test)[:,1]
print "AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro)
print"Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre) 