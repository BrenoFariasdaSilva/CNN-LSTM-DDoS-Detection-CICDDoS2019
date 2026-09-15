"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 EVALUATION METRICS AND CONFUSION MATRIX
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Computes held-out multiclass classification metrics and renders the 12-class
    confusion-matrix image produced by each CNN-LSTM CICDDoS2019 reproduction run.

    Key features include:
        - Computes accuracy plus macro and weighted precision/recall/F1 metrics.
        - Uses zero_division=0 consistently with the original implementation.
        - Saves a labeled CICDDoS2019 12-class confusion-matrix figure.

Usage:
    1. Pass integer ground-truth and prediction vectors to metrics_from_predictions().
    2. Pass a numeric confusion matrix and destination path to save_confusion().
    3. Persist returned metrics from the experiment module.

Outputs:
    - In-memory metrics dictionary.
    - Confusion-matrix PNG written to the requested destination.

TODOs:
    - None identified.

Dependencies:
    - matplotlib.
    - numpy.
    - scikit-learn.
    - cnn_lstm_ddos_detection.constants.

Assumptions & Notes:
    - Class ordering always follows PAPER_12_CLASSES.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .constants import PAPER_12_CLASSES


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    Compute the original accuracy, macro, and weighted multiclass metrics.

    :param y_true: Integer ground-truth class labels from the held-out test set.
    :param y_pred: Integer predicted class labels aligned with y_true.
    :return: Dictionary containing scalar held-out classification metrics.
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }  # Return the exact metric set used by the original implementation
