from sklearn.svm import SVR

svr_model = SVR()
svr_model.fit(X_train, y_train)