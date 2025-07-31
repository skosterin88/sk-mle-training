import os
import sys

import pickle
import json
from pathlib import Path

import numpy as np
import scipy as sp
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        y_true, pred, ax=ax, colorbar=False)
    ax.xaxis.set_tick_params(rotation=45)
    _ = ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig

def test(model: LogisticRegression, data: np.ndarray, target:np.ndarray) -> tuple[dict, Figure]:
    """
    Test the logistic regression model and return evaluation metrics.
    
    Parameters:
    model (LogisticRegression): The trained logistic regression model.
    data (np.ndarray): Feature matrix for testing.
    target (np.ndarray): True labels for testing.
    
    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    pred = model.predict(data)

    fig = conf_matrix(target, pred)
    
    return classification_report(target, pred, output_dict=True), fig

def run_evaluation(test_frame_path: Path, 
                   test_target_path: Path,
                   model_path: Path, 
                   metric_path: Path, 
                   figure_path: Path) -> None:
    """
    Run the evaluation of the logistic regression model.
    
    Parameters:
    test_frame_path (Path): Path to the test data.
    test_target_path (Path): Path to the true labels.
    model_path (Path): Path to the trained model.
    metric_path (Path): Path to save the evaluation metrics.
    figure_path (Path): Path to save the confusion matrix figure.
    
    Returns:
    None
    """

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    test_features = sp.sparse.load_npz(test_frame_path)
    test_target = np.load(test_target_path)

    report, fig = test(model, test_features, test_target)
    
    json.dump(report, open(metric_path, 'w'))
    plt.savefig(figure_path)

if __name__ == "__main__":
    
    test_frame_path = os.getcwd() + sys.argv[1]
    test_target_path = os.getcwd() + sys.argv[2]
    model_path = os.getcwd() + sys.argv[3]
    metric_path = os.getcwd() + sys.argv[4]
    figure_path = os.getcwd() + sys.argv[5]

    run_evaluation(test_frame_path, test_target_path, model_path, metric_path, figure_path)