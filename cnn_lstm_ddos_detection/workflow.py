"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 COMPLETE REPRODUCTION WORKFLOW
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Coordinates configuration persistence, accelerator setup, raw-source discovery and
    integrity checks, sampled-data caching, repeated experiment runs, and aggregation.

    Key features include:
        - Preserves the raw-data -> sample -> repeated-run execution order.
        - Reuses the existing sampled_X.npy/sampled_y.npy cache only when explicitly requested.
        - Produces per-run and aggregate experiment summaries while verifying raw-source integrity.

Usage:
    1. Parse and validate command-line arguments through cnn_lstm_ddos_detection.cli.
    2. Pass the validated argparse namespace to run_workflow().
    3. Inspect generated artifacts below the configured project-local output directory.

Outputs:
    - Configuration, environment, source snapshot, sample cache, per-run artifacts, and aggregate metrics.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - Python standard library.
    - cnn_lstm_ddos_detection configuration, constants, experiment, persistence, sampling, schema, system, and timing modules.

Assumptions & Notes:
    - The output directory is already validated to remain outside the raw dataset and inside
      the directory containing the top-level main.py file.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .constants import PAPER_12_CLASSES, PROJECT_ROOT
from .system import configure_accelerator, environment_info
from .experiment import run_experiment
from .persistence import save_base_cache_with_eta, snapshot_raw_csvs, verify_raw_snapshot
from .sampling import build_memory_safe_sample
from .schema import discover_csv_files, inspect_schemas
from .timing import format_duration


def build_config(args: argparse.Namespace) -> Config:
    """
    Build the immutable experiment Config from validated command-line arguments.

    :param args: Validated and path-normalized command-line namespace.
    :return: Immutable Config preserving the original field mapping.
    """

    return Config(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        chunksize=args.chunksize,
        rows_per_file_per_class=args.rows_per_file_per_class,
        global_class_cap=args.global_class_cap,
        prebalance_downsample=args.prebalance_downsample,
        data_seed=args.data_seed,
        runs=args.runs,
        base_seed=args.base_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        conv_filters_1=args.conv_filters_1,
        conv_filters_2=args.conv_filters_2,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        post_dense_units=args.post_dense_units,
        dropout=args.dropout,
        patience=args.patience,
        smote_k_neighbors=args.smote_k_neighbors,
        smote_generation_chunk=args.smote_generation_chunk,
        smote_neighbor_query_chunk=args.smote_neighbor_query_chunk,
        keep_inbound=args.keep_inbound,
        include_identifiers=args.include_identifiers,
        allow_cpu=args.allow_cpu,
        deterministic_ops=args.deterministic_ops,
        mixed_precision=args.mixed_precision,
        reuse_sample_cache=args.reuse_sample_cache,
        save_derived_data=not args.no_save_derived_data,
        target_accuracy=args.target_accuracy,
    )  # Preserve the original command-line-to-configuration field mapping
