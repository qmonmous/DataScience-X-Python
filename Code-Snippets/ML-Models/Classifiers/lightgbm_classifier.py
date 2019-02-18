import lightgbm as lgb
from lightgbm import LGBMClassifier

lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train)