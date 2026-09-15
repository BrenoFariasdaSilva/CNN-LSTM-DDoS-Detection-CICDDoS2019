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


def preprocess_arrays(X: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler, Dict[str, float]]:
    """
    Fit median imputation and z-score scaling on training data and transform all splits.

    :param X: Sampled real-data feature matrix before split-specific transformations.
    :param train_idx: Row indices assigned to training.
    :param val_idx: Row indices assigned to validation.
    :param test_idx: Row indices assigned to held-out testing.
    :return: Transformed train/validation/test arrays, fitted imputer/scaler, and timing metrics.
    """

    timings: Dict[str, float] = {}  # Collect preprocessing stage durations for the run manifest
    stage = time.time()  # Start split-materialization timing
    X_train = np.array(X[train_idx], dtype=np.float32, copy=True)  # Materialize an independent float32 training partition
    X_val = np.array(X[val_idx], dtype=np.float32, copy=True)  # Materialize an independent float32 validation partition
    X_test = np.array(X[test_idx], dtype=np.float32, copy=True)  # Materialize an independent float32 held-out test partition
    timings["materialize_splits_seconds"] = time.time() - stage  # Record partition materialization duration
    print(f"[PREPROCESS] materialized splits in {format_duration(timings['materialize_splits_seconds'])}")  # Report split materialization duration
    imputer = SimpleImputer(strategy="median", copy=False)  # Create the original median imputer configuration
    stage = time.time()  # Start training-imputer timing
    X_train = imputer.fit_transform(X_train).astype(np.float32, copy=False)  # Fit median values on training data and transform training data
    timings["imputer_fit_train_seconds"] = time.time() - stage  # Record train-only imputer fitting/application duration
    print(f"[PREPROCESS] fitted/applied median imputer to TRAIN in {format_duration(timings['imputer_fit_train_seconds'])}")  # Report training imputation duration
    stage = time.time()  # Start validation/test imputation timing
    X_val = imputer.transform(X_val).astype(np.float32, copy=False)  # Apply training-fitted medians to validation data
    X_test = imputer.transform(X_test).astype(np.float32, copy=False)  # Apply training-fitted medians to held-out test data
    timings["imputer_validation_test_seconds"] = time.time() - stage  # Record validation/test imputation duration
    print(f"[PREPROCESS] applied training imputer to validation/test in {format_duration(timings['imputer_validation_test_seconds'])}")  # Report non-training imputation duration
    scaler = StandardScaler(copy=False)  # Create the original z-score standardizer configuration
    stage = time.time()  # Start training-standardizer timing
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)  # Fit mean/variance on training data and standardize training data
    timings["standardizer_fit_train_seconds"] = time.time() - stage  # Record train-only standardizer fitting/application duration
    print(f"[PREPROCESS] fitted/applied z-score standardization to TRAIN in {format_duration(timings['standardizer_fit_train_seconds'])}")  # Report training standardization duration
    stage = time.time()  # Start validation/test standardization timing
    X_val = scaler.transform(X_val).astype(np.float32, copy=False)  # Apply training-fitted standardization to validation data
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)  # Apply training-fitted standardization to held-out test data
    timings["standardizer_validation_test_seconds"] = time.time() - stage  # Record validation/test standardization duration
    print(f"[PREPROCESS] applied training standardizer to validation/test in {format_duration(timings['standardizer_validation_test_seconds'])}")  # Report non-training standardization duration
    return X_train, X_val, X_test, imputer, scaler, timings  # Return transformed splits and fitted train-only preprocessing state
