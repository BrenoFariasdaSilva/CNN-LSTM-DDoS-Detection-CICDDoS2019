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
