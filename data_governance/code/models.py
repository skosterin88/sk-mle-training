import os
import sys
import pickle
from pathlib import Path

import dvc.api

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, 
                             ConfusionMatrixDisplay, 
                             classification_report)

def conf_matrix(y_true: np.ndarray, pred: np.ndarray) -> Figure:
    """
    Generate a confusion matrix figure.
    
    Parameters:
    y_true (np.ndarray): True labels.
    pred (np.ndarray): Predicted labels.
    
    Returns:
    Figure: A matplotlib figure containing the confusion matrix.
    """
    plt.ioff()  # Turn off interactive plotting

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, ax=ax, colorbar=False, cmap=False)
    ax.xaxis.set_tick_params(rotation=45)
    _ = ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig

def train(data: np.ndarray, target: np.ndarray) -> LogisticRegression:
    """
    Train a logistic regression model.
    
    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    
    Returns:
    LogisticRegression: The trained logistic regression model.
    """
    # model_params = {
    #     'multi_class': 'multinomial',
    #     'solver': 'saga',
    #     'random_state': 42,
    # }

    model_params = dvc.api.params_show()

    model_lr = LogisticRegression(**model_params['logistic_regression'])
    model_lr.fit(data, target)
    
    return model_lr

def run_train(train_frame_path: Path, 
              train_target_path: Path, 
              model_path: Path) -> None:
    """
    Train a logistic regression model and save it.
    
    Parameters:
    train_frame_path (Path): Path to the training data.
    train_target_path (Path): Path to the training target labels.
    model_path (Path): Path to save the trained model.
    """
    train_features = sp.sparse.load_npz(train_frame_path)

    train_target = np.load(train_target_path)
    
    model = train(train_features, train_target)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    
    train_frame = os.getcwd() + sys.argv[1]
    train_target = os.getcwd() + sys.argv[2]
    model_path = os.getcwd() + sys.argv[3]

    run_train(train_frame, train_target, model_path)