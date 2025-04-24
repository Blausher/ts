import pandas as pd
import numpy as np
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from source.backward_fs import backward_by_fi, select_feature_index
from source.optimizing import objective_lgbm_model


def create_mae_df(y_train, y_valid, y_test, 
                  y_train_pred, y_valid_pred, y_test_pred):
    '''
    Создает MAE train valid test dataframe для сравнения метрик
    '''
    data = [[mean_absolute_error(y_train, y_train_pred)],
            [mean_absolute_error(y_valid, y_valid_pred)],
            [mean_absolute_error(y_test, y_test_pred)]]
    columns = ['MAE']
    index = ['train', 'valid', 'test']
    df = pd.DataFrame(data, columns = columns, index = index)
    return df

def calculate_correlation_importances(x_train, y):
    correlations = x_train.corrwith(y)
    importance_df = pd.DataFrame({
        'Feature': correlations.index,
        'Importance': correlations.abs()
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    top10_features = importance_df['Feature'].iloc[:10].to_list()
    return importance_df, top10_features


def pipe(x_train: pd.DataFrame, x_valid: pd.DataFrame, x_test: pd.DataFrame,
        y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series,
        longlist: list) -> dict:
    '''
    Пайп модуля отбора признаков, опимизации гиперпараметров

    longlist - длинный список признаков для дальнейшего отбоар
    '''
    cache_dict = {}
    
    # Фильтрационный метод отбора
    importance_df, top10_features = calculate_correlation_importances(x_train, y_train)
    model_lgbm_corr = LGBMRegressor(early_stopping_rounds=100, random_state=34, n_jobs=-1, verbosity = -1)  
    model_lgbm_corr.fit(x_train[top10_features], y_train, eval_set=(x_valid[top10_features], y_valid))
    cache_dict['df_metric_lgbm_corr'] = create_mae_df(
                                      y_train = y_train,
                                      y_valid = y_valid,
                                      y_test = y_test, 
                                      y_train_pred = model_lgbm_corr.predict(x_train[top10_features]), 
                                      y_valid_pred = model_lgbm_corr.predict(x_valid[top10_features]),
                                      y_test_pred = model_lgbm_corr.predict(x_test[top10_features]))

    # LGBM FS part (слышал что LGBM хорошо работает для чего то маленького) ---------------
    short_lists, train_scores, valid_scores = backward_by_fi(x_train, y_train,
                                                                    x_valid, y_valid,
                                                                    longlist,
                                                                    percent_drop = 10, plot = False)
    selected_index = select_feature_index(valid_scores) 
    shortlist = short_lists[selected_index]

    # отобранная модель
    model_lgbm = LGBMRegressor(early_stopping_rounds=100, random_state=34, n_jobs=-1, verbosity = -1)  
    model_lgbm.fit(x_train[shortlist], y_train, eval_set=(x_valid[shortlist], y_valid))  
    cache_dict['df_metric_lgbm'] = create_mae_df(
                                      y_train = y_train,
                                      y_valid = y_valid,
                                      y_test = y_test, 
                                      y_train_pred = model_lgbm.predict(x_train[shortlist]), 
                                      y_valid_pred = model_lgbm.predict(x_valid[shortlist]),
                                      y_test_pred = model_lgbm.predict(x_test[shortlist]))

    # Lasso part --------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_valid_scaled = scaler.transform(x_valid)
    X_test_scaled = scaler.transform(x_test)

    lasso = Lasso(alpha=0.01)  # alpha обычно меньше для Lasso
    lasso.fit(X_train_scaled, y_train)

    feature_names = x_train.columns.tolist()
    nonzero_indices = lasso.coef_ != 0
    selected_features = [feature_names[i] for i, nonzero in enumerate(nonzero_indices) if nonzero]
    cache_dict['df_metric_lasso'] = create_mae_df(
                                      y_train = y_train,
                                      y_valid = y_valid,
                                      y_test = y_test, 
                                      y_train_pred = lasso.predict(X_train_scaled), 
                                      y_valid_pred = lasso.predict(X_valid_scaled),
                                      y_test_pred = lasso.predict(X_test_scaled))

    # Определение лучшего алгоритма
    df_m = pd.DataFrame(columns = ['k', 'test_metric'])
    for k in cache_dict.keys():
        table = cache_dict[k]
        metric = table.loc['test'].values[0]
        df_m.loc[len(df_m)+1] = [k, metric]

    best_algo = df_m[df_m['test_metric'].eq(df_m['test_metric'].min())]['k'].values[0]

    print('best_algo:', best_algo)
    # OPTIMIZATION part -------------------------------
    if best_algo == 'df_metric_lgbm_corr':
        # победил отбор с помощью корреляции
        objective_with_args = partial(objective_lgbm_model,
                                    X_train=x_train,
                                    y_train=y_train,
                                    X_valid=x_valid,
                                    y_valid=y_valid,
                                    selected_features=top10_features)
        
        study_model = optuna.create_study(
            study_name='lightgbm_study_model',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            direction='maximize')
        study_model.optimize(objective_with_args, n_trials=20, n_jobs=-1)
        optuna_best_params = study_model.best_params

        model = LGBMRegressor(**optuna_best_params,
                              importance_type = 'gain',
                              early_stopping_rounds = 10,
                              random_state=34, n_jobs=-1)
        model.fit(X=x_train[top10_features], y=y_train, eval_set=(x_valid[top10_features], y_valid))

        return {'model': model, 'features': top10_features}

    if best_algo == 'df_metric_lgbm':
        # победил отбор с помощью backward FS
        objective_with_args = partial(objective_lgbm_model,
                                    X_train=x_train,
                                    y_train=y_train,
                                    X_valid=x_valid,
                                    y_valid=y_valid,
                                    selected_features=shortlist)
        
        study_model = optuna.create_study(
            study_name='lightgbm_study_model',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            direction='maximize',)
        study_model.optimize(objective_with_args, n_trials=20, n_jobs=-1)
        optuna_best_params = study_model.best_params

        model = LGBMRegressor(**optuna_best_params,
                              importance_type = 'gain',
                              early_stopping_rounds = 10,
                              random_state=34, n_jobs=-1)
        model.fit(X=x_train[shortlist], y=y_train, eval_set=(x_valid[shortlist], y_valid))

        return {'model': model, 'features': shortlist}
        
    if best_algo == 'df_metric_lasso': # TODO посмотреть еще этот отбор альфы мб оптимизировать
        # победил отбор Lasso
        param_grid = {'alpha': np.logspace(-4, 1, 10)}
        lasso = Lasso(max_iter=1000)
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train_scaled, y_train)
        best_alpha = grid_search.best_params_['alpha']

        lasso = Lasso(alpha=best_alpha)  
        lasso.fit(X_train_scaled, y_train)

        return {'model': lasso, 'features': longlist,
                'X_train_scaled': X_train_scaled, 
                'X_valid_scaled': X_valid_scaled,
                'X_test_scaled': X_test_scaled
                }