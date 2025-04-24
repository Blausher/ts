import pandas as pd
import numpy as np
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
import lightgbm as lgbm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV



def objective_lgbm_model(trial, X_train, y_train, X_valid, y_valid, selected_features):
    params = {
        'random_state': 34,
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.5),
        'metric': 'binary_logloss',
        'n_jobs': 7,
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 100, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 1500),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2_000),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 1, 8),

        'early_stopping_round': 100,
        'verbosity': -1
    }

    booster = LGBMRegressor(**params)
    booster.fit(X=X_train[selected_features], y=y_train, eval_set=(X_valid[selected_features], y_valid))

    predicted_train = booster.predict(X_train[selected_features])
    predicted_valid = booster.predict(X_valid[selected_features])

    mae_train = mean_absolute_error(y_train, predicted_train)
    mae_valid = mean_absolute_error(y_valid, predicted_valid)

    return -mae_valid



