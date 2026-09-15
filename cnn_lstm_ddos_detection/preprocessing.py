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


def generate_smote_rows(class_x: np.ndarray, neighbors: np.ndarray, n_generate: int, k: int, generation_chunk: int, rng: np.random.Generator, class_name: str, generated_total: int, total_to_generate: int, smote_start: float) -> Tuple[np.ndarray, int]:
    """
    Generate one class's synthetic SMOTE rows in bounded batches.

    :param class_x: Standardized training rows belonging to one class.
    :param neighbors: Same-class neighbor-index matrix for each base row.
    :param n_generate: Number of synthetic rows required for the class.
    :param k: Effective number of eligible same-class neighbors per base row.
    :param generation_chunk: Maximum synthetic rows generated per allocation batch.
    :param rng: SMOTE random generator shared across all classes.
    :param class_name: Canonical class name used in progress messages.
    :param generated_total: Synthetic row count generated for prior classes.
    :param total_to_generate: Total synthetic rows required across all classes.
    :param smote_start: Global SMOTE start timestamp used for ETA estimation.
    :return: Concatenated synthetic rows for the class and updated global generated count.
    """

    n_current = len(class_x)  # Capture class row count for random base-sample selection
    remaining = n_generate  # Track synthetic rows still required for this class
    generated_parts: List[np.ndarray] = []  # Collect bounded synthetic batches before one class-level concatenation
    while remaining > 0:  # Continue until the class reaches the majority-class target size
        batch_n = min(remaining, generation_chunk)  # Bound the current synthetic allocation size
        base_local = rng.integers(0, n_current, size=batch_n)  # Select random base samples using the shared original RNG
        neighbor_slot = rng.integers(0, k, size=batch_n)  # Select one eligible neighbor slot for each base sample
        neighbor_local = neighbors[base_local, neighbor_slot]  # Resolve selected slots to same-class neighbor row indices
        interpolation = rng.random((batch_n, 1), dtype=np.float32)  # Draw one interpolation coefficient per synthetic row
        synthetic = class_x[base_local] + interpolation * (class_x[neighbor_local] - class_x[base_local])  # Interpolate along the base-to-neighbor segment
        generated_parts.append(synthetic.astype(np.float32, copy=False))  # Store the current float32 synthetic batch
        remaining -= batch_n  # Reduce this class's outstanding synthetic-row count
        generated_total += batch_n  # Advance the global SMOTE progress count
        elapsed = time.time() - smote_start  # Calculate elapsed global SMOTE time
        eta = eta_from_progress(generated_total, max(total_to_generate, 1), elapsed)  # Estimate remaining global SMOTE time
        print(
            f"[ETA][SMOTE] {generated_total:,}/{total_to_generate:,} "
            f"({100*generated_total/max(total_to_generate,1):.1f}%) | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)} | {class_name}"
        )  # Report synthetic generation progress using the original fields
    return np.concatenate(generated_parts, axis=0), generated_total  # Preserve original one-class concatenation behavior and progress state


def oversample_smote_class(X_train: np.ndarray, y_train: np.ndarray, class_id: int, class_name: str, target_n: int, k_neighbors: int, generation_chunk: int, neighbor_query_chunk: int, rng: np.random.Generator, generated_total: int, total_to_generate: int, smote_start: float) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, int], int]:
    """
    Oversample one training class to the multiclass SMOTE target while preserving RNG order.

    :param X_train: Standardized training feature matrix.
    :param y_train: Integer training labels aligned with X_train.
    :param class_id: Integer identifier of the class currently being processed.
    :param class_name: Canonical class name used in progress and report output.
    :param target_n: Majority-class row count used as the SMOTE target.
    :param k_neighbors: Requested number of same-class nearest neighbors.
    :param generation_chunk: Maximum synthetic rows generated per allocation batch.
    :param neighbor_query_chunk: Maximum rows queried per nearest-neighbor batch.
    :param rng: SMOTE random generator shared across classes in fixed class order.
    :param generated_total: Synthetic rows already generated for earlier classes.
    :param total_to_generate: Total synthetic rows required across all classes.
    :param smote_start: Global SMOTE start timestamp used for ETA estimation.
    :return: Optional synthetic feature/label arrays, per-class report, and updated generated count.
    """

    class_idx = np.flatnonzero(y_train == class_id)  # Locate standardized training rows belonging to the current class
    n_current = len(class_idx)  # Capture the current training class size
    n_generate = target_n - n_current  # Calculate synthetic rows required to match the majority class
    if n_generate <= 0:  # Verify if the current class already meets the target size
        class_report = {"before": n_current, "synthetic": 0, "after": n_current, "k_neighbors_used": 0}  # Record that no augmentation was required
        return None, None, class_report, generated_total  # Advance without consuming random numbers for majority classes
    if n_current < 2:  # Verify if nearest-neighbor interpolation is possible for the class
        raise RuntimeError(f"SMOTE cannot operate on class {class_name!r} with only {n_current} sample(s).")  # Reject classes that cannot form a same-class neighbor pair
    effective_k = min(int(k_neighbors), n_current - 1)  # Clamp requested neighbors to available non-self class rows
    class_x = X_train[class_idx]  # Materialize standardized rows for this class
    neighbors, neighbor_seconds = query_smote_neighbors(class_x, effective_k, class_name, neighbor_query_chunk)  # Build the exact same-class nearest-neighbor graph
    print(
        f"[SMOTE] {class_name}: exact neighbor graph ready in "
        f"{format_duration(neighbor_seconds)}; generating {n_generate:,} rows"
    )  # Preserve the original neighbor-ready status message
    class_synthetic, generated_total = generate_smote_rows(
        class_x, neighbors, n_generate, effective_k, generation_chunk, rng, class_name,
        generated_total, total_to_generate, smote_start,
    )  # Generate this class's synthetic rows with the shared SMOTE RNG state
    class_labels = np.full(len(class_synthetic), class_id, dtype=np.int16)  # Build aligned synthetic integer labels
    class_report = {"before": n_current, "synthetic": n_generate, "after": target_n, "k_neighbors_used": effective_k}  # Record per-class SMOTE details
    del neighbors, class_x  # Release class-specific nearest-neighbor memory before processing the next class
    gc.collect()  # Encourage prompt release of large class-level allocations
    return class_synthetic, class_labels, class_report, generated_total  # Return this class's synthetic data and updated progress state


def combine_smote_training(X_train: np.ndarray, y_train: np.ndarray, synthetic_x_parts: List[np.ndarray], synthetic_y_parts: List[np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Counter]:
    """
    Combine original and synthetic training rows and apply the final SMOTE shuffle.

    :param X_train: Standardized original training feature matrix.
    :param y_train: Original integer training labels.
    :param synthetic_x_parts: Synthetic feature arrays generated for minority classes.
    :param synthetic_y_parts: Synthetic integer-label arrays aligned with synthetic features.
    :param rng: Existing SMOTE random generator whose state must continue unchanged.
    :return: Shuffled resampled features, labels, and post-SMOTE class counts.
    """

    if synthetic_x_parts:  # Verify if at least one minority class required augmentation
        X_resampled = np.concatenate([X_train] + synthetic_x_parts, axis=0).astype(np.float32, copy=False)  # Append all synthetic features after original training rows
        y_resampled = np.concatenate([y_train.astype(np.int16, copy=False)] + synthetic_y_parts, axis=0)  # Append aligned synthetic labels
    else:  # Handle an already-balanced training partition
        X_resampled = X_train.astype(np.float32, copy=False)  # Preserve standardized training features without augmentation
        y_resampled = y_train.astype(np.int16, copy=False)  # Preserve integer training labels without augmentation
    order = rng.permutation(len(y_resampled))  # Draw the final SMOTE-training row permutation from the existing RNG state
    X_resampled = X_resampled[order]  # Shuffle original and synthetic training features together
    y_resampled = y_resampled[order]  # Apply the identical row permutation to training labels
    counts_after = Counter(int(value) for value in y_resampled.tolist())  # Count labels after SMOTE for audit metadata
    return X_resampled, y_resampled, counts_after  # Return the balanced and shuffled training dataset with audit counts
