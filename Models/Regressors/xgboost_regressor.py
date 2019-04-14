import xgboost as xgb
from xgboost.sklearn import XGBRegressor

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)