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


def save_confusion(confusion: np.ndarray, output_path: Path) -> None:
    """
    Render and save the labeled 12-class held-out confusion matrix.

    :param confusion: Numeric confusion matrix ordered by PAPER_12_CLASSES.
    :param output_path: Destination PNG file path.
    :return: None.
    """

    figure, axis = plt.subplots(figsize=(12, 10))  # Create the original confusion-matrix figure size
    image = axis.imshow(confusion)  # Render matrix counts with matplotlib's default image mapping
    figure.colorbar(image, ax=axis)  # Add a color scale beside the matrix
    axis.set_xticks(np.arange(len(PAPER_12_CLASSES)), labels=PAPER_12_CLASSES, rotation=45, ha="right")  # Label predicted classes on the x-axis
    axis.set_yticks(np.arange(len(PAPER_12_CLASSES)), labels=PAPER_12_CLASSES)  # Label actual classes on the y-axis
    axis.set_xlabel("Predicted")  # Preserve the original predicted-axis label
    axis.set_ylabel("Actual")  # Preserve the original actual-axis label
    axis.set_title("CICDDoS2019 12-class CNN-LSTM confusion matrix")  # Preserve the original figure title
    figure.tight_layout()  # Fit labels inside the saved figure bounds
    figure.savefig(output_path, dpi=180)  # Save the confusion matrix at the original output resolution
    plt.close(figure)  # Release matplotlib figure resources after persistence
