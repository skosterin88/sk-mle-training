import pandas as pd
from sklearn.metrics import mean_absolute_error, \
    root_mean_squared_error, \
    mean_absolute_percentage_error


def evaluate(y_pred: pd.DataFrame, y_real: pd.DataFrame, metric: str = 'rmse') -> float:
    """
    Evaluate predicted target against actual values.

    Args:
        y_pred: pd.DataFrame - predicted target values.
        y_real: pd.DataFrame - actual target values.
        metric: str - metric used for evaluation:
            'mae' for Mean Absolute Error;
            'rmse' for Root Mean Squared Error;
            'mape' for Mean Absolute Percentage Error.
    Returns:
        float - evaluation metric value.
    """

    metric_value = 10000000.0

    if metric == 'mae':
        metric_value = mean_absolute_error(y_real, y_pred)
    elif metric == 'mape':
        metric_value = mean_absolute_percentage_error(y_real, y_pred)
    else:
        metric_value = root_mean_squared_error(y_real, y_pred)
    
    return metric_value
