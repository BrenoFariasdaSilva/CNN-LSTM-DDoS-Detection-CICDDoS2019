"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 CICDDOS2019 SCHEMA AND LABEL HANDLING
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Discovers CICDDoS2019 CSV files, normalizes column and label identifiers, and
    computes the common usable feature schema across all source CSV files.

    Key features include:
        - Recursively discovers source CSV files without modifying them.
        - Preserves exact CSV header spellings required by pandas usecols.
        - Computes a stable common feature set and canonical 12-class labels.

Usage:
    1. Discover files with discover_csv_files().
    2. Inspect all headers with inspect_schemas().
    3. Canonicalize raw labels while streaming data with canonicalize_labels().

Outputs:
    - In-memory FileSchema values, feature-key lists, and canonical label series.

TODOs:
    - None identified.

Dependencies:
    - pandas.
    - cnn_lstm_ddos_detection.config.
    - cnn_lstm_ddos_detection.constants.
    - cnn_lstm_ddos_detection.timing.

Assumptions & Notes:
    - Feature selection remains the intersection of usable columns across every CSV,
      ordered according to the first discovered CSV as in the original implementation.
================================================================================
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .config import FileSchema
from .constants import BASE_LABEL_ALIASES, DEFAULT_DROP_KEYS
from .timing import eta_from_progress, format_duration


def norm_token(value: object) -> str:
    """
    Normalize an arbitrary label token to uppercase alphanumeric characters.

    :param value: Raw label value to normalize.
    :return: Normalized uppercase alphanumeric token.
    """

    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())  # Normalize labels exactly as the original implementation


def norm_column_key(name: object) -> str:
    """
    Normalize an arbitrary CSV column name to a lowercase alphanumeric key.

    :param name: Raw CSV column name to normalize.
    :return: Normalized lowercase alphanumeric column key.
    """

    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())  # Normalize headers while preserving exact names elsewhere
