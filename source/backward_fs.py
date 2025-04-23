import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy.stats import linregress, entropy

import matplotlib.pyplot as plt


def backward_by_fi(X_train, y_train, X_valid, y_valid, features, percent_drop=10, plot = False):  
    short_lists = []  
    valid_scores = []  
    train_scores = []  

    while features:  
        step = max(2, len(features) // percent_drop)  
        short_lists.append(features)  

        model = LGBMRegressor(
            early_stopping_rounds=100,  # Исправлено название параметра
            random_state=34,
            n_jobs=-1,
            verbosity = -1
        )  
        
        model.fit(
            X_train[features], y_train, 
            eval_set=(X_valid[features], y_valid)
        )  
        
        train_pred = model.predict(X_train[features])
        valid_pred = model.predict(X_valid[features])
        
        train_scores.append(mean_absolute_error(y_train, train_pred))  
        valid_scores.append(mean_absolute_error(y_valid, valid_pred))  

        # if plot:
        #     print(f'Features: {len(features)}, TRAIN R2: {train_scores[-1]:.4f}, VALID R2: {valid_scores[-1]:.4f}')  

        feature_imp = sorted(
            zip(features, model.feature_importances_),  
            key=lambda x: x[1],  
            reverse=True  
        )  

        features = [feat[0] for feat in feature_imp[:-step]]  # Удаляем наименее важные признаки

    if plot:
        feats_lens = [str(len(sl)) for sl in short_lists]

        plt.figure(figsize=(20,10))
        plt.plot(feats_lens, [i for i in train_scores], label = 'train')
        plt.plot(feats_lens, [i for i in valid_scores], label = 'valid')
        plt.xlabel('number of features')
        plt.ylabel('MAE score')
        plt.title('MAE by feature')
        plt.grid()
        plt.legend();

    return short_lists, train_scores, valid_scores, model

def select_feature_index(metrics):
    min_val = min(metrics)
    for i in reversed(range(len(metrics))):
        if metrics[i] == min_val:
            return i
    return 0  # This return is theoretically unreachable if metrics is non-empty