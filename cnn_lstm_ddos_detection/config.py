"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 CONFIGURATION MODELS
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines immutable data models shared by the CNN-LSTM CICDDoS2019 reproduction modules.
    These models centralize experiment configuration and discovered CSV schema data.

    Key features include:
        - Stores all command-line experiment configuration values.
        - Represents per-file CSV schema information.
        - Provides typed shared structures without executing pipeline stages.

Usage:
    1. Construct Config from validated command-line arguments in the workflow module.
    2. Construct FileSchema values while inspecting CICDDoS2019 CSV headers.
    3. Pass these immutable objects between pipeline modules.

Outputs:
    - None directly produced.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - Field names and semantics preserve the configuration contract of the original
      single-file implementation.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Config:
    """Store validated experiment configuration values."""

    data_dir: str
    output_dir: str
    chunksize: int
    rows_per_file_per_class: int
    global_class_cap: int
    prebalance_downsample: bool
    data_seed: int
    runs: int
    base_seed: int
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    conv_filters_1: int
    conv_filters_2: int
    kernel_size: int
    pool_size: int
    lstm_units: int
    dense_units: int
    post_dense_units: int
    dropout: float
    patience: int
    smote_k_neighbors: int
    smote_generation_chunk: int
    smote_neighbor_query_chunk: int
    keep_inbound: bool
    include_identifiers: bool
    allow_cpu: bool
    deterministic_ops: bool
    mixed_precision: bool
    reuse_sample_cache: bool
    save_derived_data: bool
    target_accuracy: float


@dataclass(frozen=True)
class FileSchema:
    """Store the discovered label and feature-column mapping for one CSV file."""

    path: Path
    label_column: str
    columns_by_key: Mapping[str, str]
