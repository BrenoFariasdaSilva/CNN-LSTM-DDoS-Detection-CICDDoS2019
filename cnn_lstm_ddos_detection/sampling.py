"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 MEMORY-SAFE CICDDOS2019 SAMPLING
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Streams every discovered CICDDoS2019 CSV in bounded pandas chunks and supports
    bounded per-file/per-class sampling or an explicit retain-all mode before global capping.

    Key features include:
        - Reads raw CSV files without loading the complete corpus into memory.
        - Preserves uniform bounded per-file/per-class priority sampling behavior.
        - Supports zero as an explicit no-cap mode that retains every target row.
        - Builds the sampled in-memory dataset and sampling audit report.

Usage:
    1. Inspect source schemas before calling build_memory_safe_sample().
    2. Supply the configured chunk size and sampling caps.
    3. Use the returned arrays as the immutable real-data basis for repeated runs.

Outputs:
    - Sampled feature and integer-label NumPy arrays in memory.
    - Sampling audit report returned to the caller.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - cnn_lstm_ddos_detection.config.
    - cnn_lstm_ddos_detection.constants.
    - cnn_lstm_ddos_detection.schema.
    - cnn_lstm_ddos_detection.timing.

Assumptions & Notes:
    - Raw CSV files are opened for reading only and are never modified by this module.
    - Random-number generator call ordering preserves the original bounded-sampling implementation.
    - Retain-all mode avoids unnecessary random priorities and concatenates each class once per source file.
================================================================================
"""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FileSchema
from .constants import (
    OFFICIAL_FIRST_DAY_ATTACKS,
    OFFICIAL_SECOND_DAY_ATTACKS,
    OMITTED_DEFAULT_CLASSES,
    PAPER_12_CLASSES,
)
from .schema import canonicalize_labels
from .timing import ByteProgress, create_byte_progress


def priority_reservoir_merge(current_x: Optional[np.ndarray], current_p: Optional[np.ndarray], new_x: np.ndarray, new_p: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge priority-reservoir candidates and retain at most the configured cap.

    :param current_x: Existing retained feature rows or None for an empty reservoir.
    :param current_p: Existing retained random priorities or None for an empty reservoir.
    :param new_x: New candidate feature rows.
    :param new_p: Random priorities corresponding to the new candidate rows.
    :param cap: Maximum number of rows retained in the reservoir.
    :return: Retained feature rows and their retained priorities.
    """

    if current_x is None:  # Verify if the per-class reservoir is currently empty
        features = new_x  # Initialize the feature reservoir with the new candidates
        priorities = new_p  # Initialize the priority reservoir with the new priorities
    else:  # Handle a reservoir that already contains sampled candidates
        features = np.concatenate((current_x, new_x), axis=0)  # Merge existing and new feature candidates
        priorities = np.concatenate((current_p, new_p), axis=0)  # Merge existing and new priority values
    if len(priorities) > cap:  # Verify if the merged reservoir exceeds its configured capacity
        indices = np.argpartition(priorities, cap - 1)[:cap]  # Select the rows with the smallest random priorities
        features = features[indices]  # Retain only selected feature rows
        priorities = priorities[indices]  # Retain priorities aligned with selected feature rows
    return features, priorities  # Return the bounded reservoir state


def update_observed_counts(labels: pd.Series, observed: Counter, omitted: Counter) -> None:
    """
    Update per-file target and omitted-class counts from canonical chunk labels.

    :param labels: Canonicalized labels for one streamed chunk.
    :param observed: Mutable counter for target-class observations.
    :param omitted: Mutable counter for recognized labels outside the target 12 classes.
    :return: None.
    """

    for label, count in labels.value_counts(dropna=True).items():  # Count each recognized canonical label in the current chunk
        canonical = str(label)  # Normalize the pandas label value to a regular string
        if canonical in PAPER_12_CLASSES:  # Verify if the label belongs to the target 12-class reconstruction
            observed[canonical] += int(count)  # Accumulate target-class observations
        else:  # Handle recognized dataset classes intentionally omitted from the reconstruction
            omitted[canonical] += int(count)  # Accumulate omitted-class observations


def build_numeric_chunk(chunk: pd.DataFrame, schema: FileSchema, feature_keys: Sequence[str], labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert target rows from one CSV chunk into numeric feature and label arrays.

    :param chunk: Raw pandas chunk read from one source CSV.
    :param schema: Exact schema metadata for the source CSV.
    :param feature_keys: Ordered normalized feature keys shared by all CSV files.
    :param labels: Canonicalized labels aligned with the raw chunk.
    :return: Float32 feature matrix and canonical object-label vector for target rows.
    """

    mask = labels.isin(PAPER_12_CLASSES)  # Identify rows belonging to the target 12 classes
    if not mask.any():  # Verify if the current chunk contains no target rows
        return np.empty((0, len(feature_keys)), dtype=np.float32), np.empty((0,), dtype=object)  # Return empty aligned arrays without changing RNG state
    actual_feature_columns = [schema.columns_by_key[key] for key in feature_keys]  # Resolve normalized keys to exact source CSV headers
    part = chunk.loc[mask, actual_feature_columns]  # Select only target rows and ordered feature columns
    rename_to_key = {schema.columns_by_key[key]: key for key in feature_keys}  # Build exact-header to normalized-key mapping
    part = part.rename(columns=rename_to_key)  # Normalize DataFrame column labels across source files
    part = part.loc[:, list(feature_keys)]  # Reassert the common feature order explicitly
    numeric_columns: List[np.ndarray] = []  # Collect converted feature columns before column stacking
    for key in feature_keys:  # Convert every common feature independently to preserve original coercion behavior
        series = pd.to_numeric(part[key], errors="coerce")  # Coerce malformed feature values to missing values
        values = series.to_numpy(dtype=np.float32, copy=False)  # Materialize the feature as float32
        values[~np.isfinite(values)] = np.nan  # Convert positive and negative infinity to missing values
        numeric_columns.append(values)  # Preserve feature-column order for stacking
    features = np.column_stack(numeric_columns).astype(np.float32, copy=False)  # Build the chunk feature matrix
    target_labels = labels.loc[mask].to_numpy(dtype=object)  # Materialize canonical target labels aligned with features
    return features, target_labels  # Return numeric target rows for reservoir sampling
