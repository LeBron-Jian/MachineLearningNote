def random_forest_train(feature_data, label_data, test_feature, submitfile):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
 
    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.23)
    params = {
        'n_estimators': 70,
        'max_depth': 13,
        'min_samples_split': 10,
        'min_samples_leaf': 10,
        'max_features': 7
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    print(RMSE)
 
    submit = pd.read_csv(submitfile)
    submit['y'] = model.predict(test_feature)
    submit.to_csv('my_random_forest_prediction1.csv', index=False)