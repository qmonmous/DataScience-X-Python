from catboost import CatBoostRegressor

cb_model = CatBoostRegressor()
cb_model.fit(X_train, y_train)