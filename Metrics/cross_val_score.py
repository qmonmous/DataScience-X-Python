from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(rf, X_train, y_train, cv=5)
np.mean(cv_score)