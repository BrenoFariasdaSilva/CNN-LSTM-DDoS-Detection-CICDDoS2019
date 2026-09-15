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


def discover_csv_files(root: Path) -> List[Path]:
    """
    Discover all CSV files recursively below the raw dataset root.

    :param root: Root directory of the CICDDoS2019 dataset.
    :return: Sorted list of discovered CSV file paths.
    """

    files = sorted(path for path in root.rglob("*.csv") if path.is_file())  # Discover source CSV files deterministically
    if not files:  # Verify if recursive discovery returned no usable CSV files
        raise FileNotFoundError(f"No .csv files were found recursively under {root}")  # Reject an invalid or incomplete dataset root
    return files  # Return all source CSV paths in deterministic lexical order


def infer_label_column(columns: Sequence[str]) -> str:
    """
    Identify and return the exact CSV header used for the label column.

    :param columns: Exact column headers read from one CSV file.
    :return: Exact original label-column header used in the CSV.
    """

    mapping = {str(column).strip().lower(): str(column) for column in columns}  # Map normalized candidates back to exact CSV headers
    for candidate in ("label", "class", "attack", "target"):  # Check supported label-column names in the original priority order
        if candidate in mapping:  # Verify if the current label candidate exists in this CSV
            return mapping[candidate]  # Return the exact header so pandas usecols can match whitespace precisely
    raise ValueError(f"Could not find label column in columns: {list(columns)[:20]}")  # Reject CSV files without a recognizable target column


def build_drop_keys(include_identifiers: bool, keep_inbound: bool) -> Set[str]:
    """
    Build the normalized feature-key exclusion set for schema inspection.

    :param include_identifiers: Whether Flow ID, IP, and timestamp identifiers should be retained.
    :param keep_inbound: Whether the Inbound feature should be retained.
    :return: Normalized feature keys that must be excluded.
    """

    drop_keys = set(DEFAULT_DROP_KEYS)  # Start from the original default feature exclusions
    if include_identifiers:  # Verify if identifier-style columns were explicitly requested
        drop_keys -= {"flowid", "sourceip", "destinationip", "srcip", "dstip", "timestamp"}  # Restore identifier keys to the candidate schema
    if keep_inbound:  # Verify if the collection-specific Inbound feature was explicitly requested
        drop_keys.discard("inbound")  # Restore Inbound to the candidate schema
    return drop_keys  # Return the effective feature exclusion set


def inspect_single_schema(path: Path, drop_keys: Set[str], display_names: Dict[str, str]) -> Tuple[FileSchema, Set[str]]:
    """
    Inspect one CSV header and derive its usable normalized feature keys.

    :param path: CSV file whose header should be inspected.
    :param drop_keys: Normalized feature keys excluded from model inputs.
    :param display_names: Mutable mapping that records the first observed display name for each key.
    :return: File schema and usable normalized feature-key set for the CSV.
    """

    header = pd.read_csv(path, nrows=0)  # Read only the CSV header to avoid loading source records
    columns = [str(column) for column in header.columns]  # Preserve exact CSV header spellings
    label = infer_label_column(columns)  # Identify the exact target-column header
    columns_by_key: Dict[str, str] = {}  # Prepare normalized-to-exact column mapping
    for column in columns:  # Process headers in their original CSV order
        key = norm_column_key(column)  # Normalize the current column for cross-file comparison
        if not key or key.startswith("unnamed"):  # Verify if the normalized column is empty or an unnamed index artifact
            continue  # Exclude unusable unnamed/index columns
        columns_by_key.setdefault(key, column)  # Preserve the first exact header for each normalized key
        display_names.setdefault(key, column.strip())  # Preserve the first clean display name across files
    label_key = norm_column_key(label)  # Normalize the label key so it can be excluded from features
    candidate_keys = set(columns_by_key) - {label_key} - drop_keys  # Compute usable features for this individual CSV
    schema = FileSchema(path=path, label_column=label, columns_by_key=columns_by_key)  # Capture exact per-file schema information
    return schema, candidate_keys  # Return schema metadata and this CSV's candidate feature set
