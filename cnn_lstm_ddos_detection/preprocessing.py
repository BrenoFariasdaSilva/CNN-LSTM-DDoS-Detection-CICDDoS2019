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


def query_smote_neighbors(class_x: np.ndarray, k: int, class_name: str, neighbor_query_chunk: int) -> Tuple[np.ndarray, float]:
    """
    Build and query the exact same-class k-nearest-neighbor graph in bounded chunks.

    :param class_x: Standardized training rows belonging to one target class.
    :param k: Effective number of same-class neighbors excluding each sample itself.
    :param class_name: Canonical class name used in progress messages.
    :param neighbor_query_chunk: Maximum number of class rows queried per nearest-neighbor batch.
    :return: Neighbor-index matrix and total graph-construction duration in seconds.
    """

    neighbor_started = time.time()  # Start nearest-neighbor graph timing before estimator construction
    estimator = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)  # Create the original exact brute-force neighbor estimator
    estimator.fit(class_x)  # Fit the estimator to this class's standardized training rows only
    n_current = len(class_x)  # Capture current class size for allocation and progress
    query_chunk = min(int(neighbor_query_chunk), n_current)  # Bound query size to the actual class row count
    neighbors = np.empty((n_current, k), dtype=np.int32)  # Allocate exact same-class neighbor indices excluding self
    query_started = time.time()  # Start bounded neighbor-query timing
    for query_start in range(0, n_current, query_chunk):  # Query class rows in bounded batches to limit memory use
        query_end = min(query_start + query_chunk, n_current)  # Clamp the current query endpoint to the class size
        batch_neighbors = estimator.kneighbors(class_x[query_start:query_end], return_distance=False)[:, 1:]  # Query exact neighbors and remove the self neighbor
        neighbors[query_start:query_end] = batch_neighbors.astype(np.int32, copy=False)  # Store bounded-query neighbor indices
        elapsed_query = time.time() - query_started  # Calculate elapsed neighbor-query time
        eta_query = eta_from_progress(query_end, n_current, elapsed_query)  # Estimate remaining neighbor-query time
        print(
            f"[ETA][SMOTE-KNN] {class_name}: {query_end:,}/{n_current:,} "
            f"({100*query_end/n_current:.1f}%) | elapsed={format_duration(elapsed_query)} | "
            f"ETA={format_duration(eta_query)}"
        )  # Report exact-neighbor query progress
    return neighbors, time.time() - neighbor_started  # Return the complete neighbor graph and construction duration
