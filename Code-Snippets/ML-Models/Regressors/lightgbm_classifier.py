import lightgbm as lgb
from lightgbm import LGBMRegressor

lgbm_model = lgb.LGBMRegressor()
lgbm_model.fit(X_train, y_train)