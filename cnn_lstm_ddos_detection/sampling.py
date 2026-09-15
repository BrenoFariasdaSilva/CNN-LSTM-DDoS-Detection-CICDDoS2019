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
