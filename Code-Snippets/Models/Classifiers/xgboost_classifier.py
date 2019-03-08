import xgboost as xgb
from xgboost.sklearn import XGBClassifier

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)