import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import pandas as pd
from xgboost import XGBRegressor

from constants import DATA_DIR
from data_loader import read_data
from data_preprocessing import create_features, train_test_split
from data_modeling import create_X_y, train_model, get_model_predictions
from evaluation import evaluate

from typing import Any, Dict, List, Tuple


def run(filename: str, 
        start_date_test: str, 
        features: List[str], 
        target: str, 
        model_name: str = 'xgb', 
        params: Dict = {}, 
        metric: str = 'rmse') -> Tuple[pd.DataFrame, float]:
    """
    Run model end-to-end.

    Args:
        filename: str - name of file containing source data
        start_date_test: str - date from which training dataset ends and test dataset starts
        features: List[str] - features used in the model
        target: str - target predicted by the model
        model_name: str - model used for generating predictions
        params: Dict - model hyperparameters
        metric: str - name of the metric to evaluate the model.
    
    Returns:
        pd.DataFrame - prediction values
        float - metric value
    """
    filepath = str(Path(DATA_DIR / filename))
    df = read_data(filepath)

    df = create_features(df)

    df_train, df_test = train_test_split(df, start_date_test)
    X_train, y_train, X_test, y_test = create_X_y(df_train, df_test, features=features, target=target)

    if model_name == 'xgb':
        untrained_model = XGBRegressor

    model = train_model(untrained_model, X_train, y_train, params)

    y_pred = get_model_predictions(model, X_test)

    metric_value = evaluate(y_pred, y_test, metric=metric)

    return y_pred, metric_value


if __name__ == '__main__':

    filename = 'PJME_hourly.csv'
    start_date_test = '01-01-2015'

    features = ['dayofyear', 
                'hour', 
                'dayofweek', 
                'quarter', 
                'month', 
                'year']
    target = 'PJME_MW'

    params = {
        'base_score':0.5, 
        'booster':'gbtree',    
        'n_estimators':1000,
        'objective':'reg:linear',
        'max_depth':3,
        'learning_rate':0.01
    }

    model_name = 'xgb'
    metric = 'rmse'

    y_pred, metric_value = run(filename, 
                               start_date_test, 
                               features, 
                               target, 
                               model_name=model_name, 
                               params=params,
                               metric=metric)
    
    print(f'{metric.upper()} == {metric_value}')