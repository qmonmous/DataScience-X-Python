from sklearn.svm import SVR

svc_model = SVR()
svc_model = svc_model.fit(X_train, y_train)