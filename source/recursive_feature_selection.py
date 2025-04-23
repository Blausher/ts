import inspect  

import pandas as pd  
import numpy as np  

from typing import Any, Union, Optional  
from copy import deepcopy  

def recursive_feature_selection_function(  
    selector: Any,  
    model_for_selection: Any,  
    X: Union[pd.DataFrame, np.ndarray],  
    y: Union[pd.Series, np.ndarray],  
    *,  
    sample_for_selection: str = 'valid',  
    plot_feature_selected: bool = False,  
    bigger_metric_is_better: bool = True,  
    max_n_feats: Optional[int] = 25,  
    figsize: tuple = (20, 20),  
    **kwargs  
    ) -> list:

    need_cols_result = set(
    ['num_features', 'features_set', 'train_metric_mean', 'val_metric_mean'])

    # Проверка типов данных для обучения
    # x = check_x_data(x)
    # y = check_y_data(y)
    
    # Проверка корректности задания входных параметров
    # check_params_main_func(max_n_feats, plot_feature_selected,
    #     bigger_metric_is_better)
    
    # Инициализация модели
    if inspect.isclass(model_for_selection):
        model = model_for_selection(**kwargs['model_params'])
    else:
        model = deepcopy(model_for_selection)
    
    # Инициализация объекта отбора фичей
    if inspect.isclass(selector):
        feature_selector = selector(model, **kwargs['selector_params'])
    else:
        feature_selector = deepcopy(selector)
    
    # Проверка наличия нужного метода вычислений
    methods = dir(feature_selector)
    
    assert 'fit_compute' in methods, \
        f'У объекта {feature_selector} отсутствует метод fit_compute'
    
    # Основные вычисления
    report = feature_selector.fit_compute(x, y)
    
    # Проверка корректности типа выходных данных
    assert isinstance(report, pd.DataFrame), \
        f'Некорректный тип выхода объекта {feature_selector}. Ожидаемый тип - pd.DataFrame, полученный - {type(report)}'
    
    fact_cols_result = set(report.columns)
    missing_cols = need_cols_result - fact_cols_result
    
    assert missing_cols == set([]), \
        f'В {report} отсутствуют колонки {missing_cols}'
    
    # Выбор финальных фичей
    sample_col = 'val_metric_mean' if sample_for_selection == 'valid' else 'train_metric_mean'
    
    # Поиск лучшей метрики и расчет разницы между всеми метриками и лучшей
    better_value_metric = report[sample_col].max() if bigger_metric_is_better \
        else report[sample_col].min()
    
    report['diff_metric'] = abs(report[sample_col] - better_value_metric)
    
    # Выбор результата в зависимости от ограничения числа признаков
    # Если ограничение по числу признаков не задано, оставляем исходный результат
    # Иначе выбираем subsample с учетом ограничения числа признаков
    sample_check = report.copy() if max_n_feats is None \
        else report[report['num_features'] <= max_n_feats].copy()
    
    # Итоговый результат
    final_result = sample_check.loc[sample_check['diff_metric'].idxmin()]
    final_features_set = final_result['features_set']

    final_report_table = report.style.set_properties(**{'background-color': 'gren'}, subset=pd.IndexSlice[final_result.name, final_result.index])

    if plot_feature_selected:
        feature_selector.plot(figsize=figsize)

    return list(final_features_set), final_report_table

    # def check_params_main_func(max_n_feats, plot_feature_selected,
    #     bigger_metric_is_better):

        








        


    