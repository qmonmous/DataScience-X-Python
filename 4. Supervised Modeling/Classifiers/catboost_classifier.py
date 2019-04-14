from catboost import CatBoostClassifier

cb_model = CatBoostClassifier()
cb_model.fit(X_train, y_train)