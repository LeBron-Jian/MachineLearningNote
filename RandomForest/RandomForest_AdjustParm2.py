def random_forest_parameter_tuning2(feature_data, label_data, test_feature):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
 
    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    param_test2 = {
        'max_depth': range(3, 14, 2),
        'min_samples_split': range(50, 201, 20)
    }
    model = GridSearchCV(estimator=RandomForestRegressor(
        n_estimators=70, min_samples_leaf=20, max_features='sqrt', oob_score=True,
        random_state=10), param_grid=param_test2, cv=5
    )
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    return model.best_score_, model.best_params_