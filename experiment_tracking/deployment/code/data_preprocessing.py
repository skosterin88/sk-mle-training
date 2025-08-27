import pandas as pd
from typing import Tuple


def train_test_split(df: pd.DataFrame, start_date_test: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Split dataset into subsets for training and testing.

    Args:
        df: pd.DataFrame - source dataset
        start_date_test: str - date from which test dataset begins.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] - train and test subsets.
    """

    train = df.loc[df.index < start_date_test]
    test = df.loc[df.index >= start_date_test]

    return train, test

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features based on time series index.
    
    Args:
        df: pd.DataFrame - source dataset.

    Returns:
        pd.DataFrame - dataset with created features.
    """
    df_with_features = df.copy()
    df_with_features['hour'] = df_with_features.index.hour
    df_with_features['dayofweek'] = df_with_features.index.dayofweek
    df_with_features['quarter'] = df_with_features.index.quarter
    df_with_features['month'] = df_with_features.index.month
    df_with_features['year'] = df_with_features.index.year
    df_with_features['dayofyear'] = df_with_features.index.dayofyear
    df_with_features['dayofmonth'] = df_with_features.index.day
    df_with_features['weekofyear'] = df_with_features.index.isocalendar().week
    
    return df_with_features