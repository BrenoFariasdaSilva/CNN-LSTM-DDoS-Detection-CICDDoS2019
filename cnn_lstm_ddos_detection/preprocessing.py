"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 SPLITTING, PREPROCESSING, AND TRAINING-ONLY SMOTE
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Implements the 70/15/15 stratified split, train-fitted median imputation and
    z-score standardization, and the custom memory-aware multiclass SMOTE procedure.

    Key features include:
        - Splits sampled real data into 70% train, 15% validation, and 15% held-out test.
        - Fits imputation and scaling exclusively on the training partition.
        - Applies exact k-nearest-neighbor SMOTE exclusively to standardized training data.

Usage:
    1. Split integer labels with split_indices().
    2. Transform feature arrays with preprocess_arrays().
    3. Oversample only training data with apply_smote().

Outputs:
    - In-memory transformed arrays, fitted preprocessing objects, timings, and SMOTE report.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - scikit-learn.
    - cnn_lstm_ddos_detection.constants.
    - cnn_lstm_ddos_detection.timing.

Assumptions & Notes:
    - Validation and test partitions never enter SMOTE and never fit preprocessing state.
    - SMOTE random-number and nearest-neighbor behavior preserves the original implementation.
================================================================================
"""

from __future__ import annotations

import gc
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .constants import PAPER_12_CLASSES
from .timing import eta_from_progress, format_duration


def split_indices(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified 70% training, 15% validation, and 15% test row indices.

    :param y: Integer class labels for the sampled real-data dataset.
    :param seed: Random seed used by both stratified split operations.
    :return: Training, validation, and held-out test index arrays.
    """

    indices = np.arange(len(y), dtype=np.int64)  # Create stable source-row indices for split auditing
    train_idx, temporary_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )  # Allocate 70% training rows and 30% temporary validation/test rows
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=y[temporary_idx],
    )  # Divide the temporary partition equally into 15% validation and 15% test
    return train_idx, validation_idx, test_idx  # Return the three disjoint stratified partitions
