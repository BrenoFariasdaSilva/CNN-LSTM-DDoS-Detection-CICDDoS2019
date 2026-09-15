"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 DATA AND METADATA PERSISTENCE
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Persists sampled and transformed NumPy arrays, preprocessing manifests, raw-source
    snapshots, and the reusable base-sample cache used by the reproduction workflow.

    Key features include:
        - Saves large NumPy arrays incrementally through open_memmap with ETA reporting.
        - Persists every transformed train/validation/test dataset artifact per run.
        - Snapshots and verifies raw CSV size/mtime metadata to detect source modification.

Usage:
    1. Use save_npy_with_eta() for large NumPy arrays.
    2. Use persist_generated_dataset() after preprocessing and training-only SMOTE.
    3. Use snapshot_raw_csvs() and verify_raw_snapshot() around the complete workflow.

Outputs:
    - NumPy .npy arrays and JSON manifests below the configured output directory.
    - Base sampled_X.npy and sampled_y.npy cache files when requested by the workflow.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - Python standard library.
    - cnn_lstm_ddos_detection.timing.

Assumptions & Notes:
    - Source raw CSV files are inspected but never written by this module.
================================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .timing import eta_from_progress, format_duration


def save_npy_with_eta(array: np.ndarray, path: Path, label: str, chunk_rows: int = 100_000) -> None:
    """
    Persist a NumPy array incrementally while reporting save progress and ETA.

    :param array: NumPy array to persist.
    :param path: Destination .npy path.
    :param label: Human-readable label shown in progress messages.
    :param chunk_rows: Maximum number of first-axis rows copied per write step.
    :return: None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists before creating the array file
    started = time.time()  # Start save-duration measurement
    memmap = np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)  # Create the destination .npy file without a second full-array copy
    total = len(array) if array.ndim else 1  # Calculate first-axis progress units while supporting scalar arrays
    if array.ndim == 0:  # Verify if the source array is scalar
        memmap[...] = array  # Persist the scalar value directly
    else:  # Handle arrays with a first dimension that can be copied incrementally
        for start in range(0, total, chunk_rows):  # Copy bounded row chunks to the memory-mapped destination
            end = min(start + chunk_rows, total)  # Clamp the current chunk endpoint to the array size
            memmap[start:end] = array[start:end]  # Persist the current row slice
            elapsed = time.time() - started  # Calculate elapsed save time
            eta = eta_from_progress(end, total, elapsed)  # Estimate remaining save time
            print(
                f"[ETA][SAVE] {label}: {end:,}/{total:,} ({100*end/total:.1f}%) | "
                f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
            )  # Report save progress using the original fields
    memmap.flush()  # Flush pending memory-mapped writes to the destination file
    del memmap  # Release the memory-map object after persistence completes


def persist_generated_dataset(generated_dir: Path, X_train_smote: np.ndarray, y_train_smote: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, manifest: Mapping[str, object]) -> None:
    """
    Persist one run's transformed train, validation, test, index, and manifest artifacts.

    :param generated_dir: Destination directory for transformed dataset artifacts.
    :param X_train_smote: Standardized training features after training-only SMOTE.
    :param y_train_smote: Integer training labels after training-only SMOTE.
    :param X_val: Standardized validation features without augmentation.
    :param y_val: Integer validation labels without augmentation.
    :param X_test: Standardized held-out test features without augmentation.
    :param y_test: Integer held-out test labels without augmentation.
    :param train_idx: Original sampled-dataset row indices assigned to training.
    :param val_idx: Original sampled-dataset row indices assigned to validation.
    :param test_idx: Original sampled-dataset row indices assigned to held-out test.
    :param manifest: Preprocessing metadata describing transformations for the run.
    :return: None.
    """

    generated_dir.mkdir(parents=True, exist_ok=True)  # Ensure the per-run transformed-data directory exists
    print(f"[DERIVED] Saving transformed dataset under {generated_dir}")  # Report transformed-data persistence location
    save_npy_with_eta(X_train_smote, generated_dir / "X_train_standardized_smote.npy", "X_train_standardized_smote")  # Persist SMOTE training features
    save_npy_with_eta(y_train_smote, generated_dir / "y_train_smote.npy", "y_train_smote")  # Persist SMOTE training labels
    save_npy_with_eta(X_val, generated_dir / "X_validation_standardized.npy", "X_validation_standardized")  # Persist untouched validation features after train-fitted transformations
    save_npy_with_eta(y_val, generated_dir / "y_validation.npy", "y_validation")  # Persist validation labels
    save_npy_with_eta(X_test, generated_dir / "X_test_standardized.npy", "X_test_standardized")  # Persist held-out test features after train-fitted transformations
    save_npy_with_eta(y_test, generated_dir / "y_test.npy", "y_test")  # Persist held-out test labels
    save_npy_with_eta(train_idx, generated_dir / "train_indices.npy", "train_indices")  # Persist training row indices
    save_npy_with_eta(val_idx, generated_dir / "validation_indices.npy", "validation_indices")  # Persist validation row indices
    save_npy_with_eta(test_idx, generated_dir / "test_indices.npy", "test_indices")  # Persist test row indices
    (generated_dir / "preprocessing_manifest.json").write_text(
        json.dumps(dict(manifest), indent=2), encoding="utf-8"
    )  # Persist the preprocessing manifest in the original JSON format


def snapshot_raw_csvs(csv_files: Sequence[Path], root: Path) -> Dict[str, Dict[str, int]]:
    """
    Snapshot raw CSV size and nanosecond modification-time metadata using relative paths.

    :param csv_files: Source CSV files whose metadata should be captured.
    :param root: Raw dataset root used to derive stable relative snapshot keys.
    :return: Mapping from source-relative path to size and modification-time metadata.
    """

    snapshot: Dict[str, Dict[str, int]] = {}  # Build the raw-source integrity snapshot incrementally
    for path in csv_files:  # Capture metadata for every discovered source CSV
        stat_result = path.stat()  # Read source size and modification timestamp without opening it for writing
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)  # Preserve the original relative-key behavior when possible
        snapshot[relative] = {"size": int(stat_result.st_size), "mtime_ns": int(stat_result.st_mtime_ns)}  # Record the exact integrity fields used originally
    return snapshot  # Return the complete raw-source metadata snapshot


def verify_raw_snapshot(before: Mapping[str, Mapping[str, int]], csv_files: Sequence[Path], root: Path) -> None:
    """
    Verify raw CSV metadata is unchanged after the reproduction workflow finishes.

    :param before: Source metadata snapshot captured before processing.
    :param csv_files: Source CSV files to snapshot again after processing.
    :param root: Raw dataset root used for stable relative snapshot keys.
    :return: None.
    """

    after = snapshot_raw_csvs(csv_files, root)  # Capture a fresh source snapshot after all experiment work
    if dict(before) != after:  # Verify if any source path, size, or modification timestamp changed
        changed = sorted(set(before) | set(after))  # Build the union of snapshot keys to locate differences
        changed = [key for key in changed if before.get(key) != after.get(key)]  # Retain only entries whose metadata changed
        raise RuntimeError(
            "Raw dataset integrity verification failed: source CSV metadata changed during execution: "
            + ", ".join(changed[:20])
        )  # Reject any run that violates the raw read-only integrity guarantee
    print("[RAW-INTEGRITY] Verified: raw CICDDoS2019 CSV sizes/mtimes are unchanged.")  # Report successful post-run source verification


def save_base_cache_with_eta(X: np.ndarray, y: np.ndarray, cache_x: Path, cache_y: Path) -> None:
    """
    Persist the sampled real-data feature and label caches with progress reporting.

    :param X: Sampled real-data feature matrix.
    :param y: Sampled integer-label vector.
    :param cache_x: Destination .npy path for sampled features.
    :param cache_y: Destination .npy path for sampled labels.
    :return: None.
    """

    save_npy_with_eta(X, cache_x, "sampled_X")  # Persist sampled real-data features using incremental .npy writing
    save_npy_with_eta(y, cache_y, "sampled_y")  # Persist sampled integer labels using incremental .npy writing
