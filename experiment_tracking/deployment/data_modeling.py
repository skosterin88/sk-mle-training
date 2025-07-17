import pandas as pd

from typing import Any, Dict, List, Tuple

def create_X_y(df_train: pd.DataFrame, 
               df_test: pd.DataFrame, 
               features: List[str], 
               target: str
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates features-only dataframes (X) and target-only dataframes (y)
    for train and test sets.

    Args:
        df_train: pd.DataFrame - train dataset.
        df_test: pd.DataFrame - test dataset.
        features: List[str] - list of features.
        target: str - target variable.

    Returns:
        Tuple[pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame] - X and y datasets created from train and test datasets.
    """
    X_train = df_train[features]
    y_train = df_train[target]

    X_test = df_test[features]
    y_test = df_test[target]

    return X_train, y_train, X_test, y_test
    
def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.DataFrame, params: Dict) -> Any:
    """
    Train model using specified hyperparams on the specified training dataset.

    Args:
        model: Any - model to train.
        X_train: pd.DataFrame - feature values in the training dataset.
        y_train: pd.DataFrame - target values in the training dataset.
        params: Dict - model hyperparameters.

    Returns:
        Any - model fitted on the training dataset.
    """

    trained_model = model(**params).fit(X_train, y_train)

    return trained_model

def get_model_predictions(trained_model: Any, X_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Get predictions provided by the trained model based on the features of the prediction dataset.

    Args: 
        trained_model: Any - model previously fitted on the train dataset.
        X_pred: pd.DataFrame - feature values of the prediction dataset.

    Returns: 
        pd.DataFrame - predicted target values.
    """

    y_pred = trained_model.predict(X_pred)

    return y_pred
