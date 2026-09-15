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


def persist_configuration(cfg: Config, output_dir: Path) -> None:
    """
    Persist the resolved experiment configuration before runtime execution begins.

    :param cfg: Immutable experiment configuration.
    :param output_dir: Validated generated-output directory.
    :return: None.
    """

    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")  # Persist configuration using the original JSON representation


def persist_source_inventory(csv_files: Sequence[Path], data_dir: Path, output_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Persist the pre-run raw-source snapshot and recursive CSV file inventory.

    :param csv_files: Ordered source CSV files discovered recursively.
    :param data_dir: Raw dataset root used for relative source paths.
    :param output_dir: Generated-output directory receiving inventory artifacts.
    :return: Raw-source metadata snapshot used later for integrity verification.
    """

    raw_snapshot = snapshot_raw_csvs(csv_files, data_dir)  # Capture source size/mtime metadata before any experiment processing
    (output_dir / "raw_dataset_snapshot_before.json").write_text(
        json.dumps(raw_snapshot, indent=2), encoding="utf-8"
    )  # Persist the pre-run source integrity snapshot
    (output_dir / "csv_files.txt").write_text(
        "\n".join(str(path.relative_to(data_dir)) for path in csv_files), encoding="utf-8"
    )  # Persist the original recursive source-file inventory format
    return raw_snapshot  # Return the snapshot for post-run integrity comparison


def load_cached_sample(cache_x: Path, cache_y: Path, cache_features: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the reusable sampled real-data arrays and feature names from project output.

    :param cache_x: Existing sampled feature-array .npy path.
    :param cache_y: Existing sampled integer-label .npy path.
    :param cache_features: Existing feature-name JSON path.
    :return: Sampled feature matrix, label vector, and ordered readable feature names.
    """

    print("[DATA] Reusing cached sampled dataset from project output directory.")  # Report explicit reuse of the existing sampled dataset cache
    features = np.load(cache_x, mmap_mode=None).astype(np.float32, copy=False)  # Load sampled features exactly as the original cache path does
    labels = np.load(cache_y, mmap_mode=None).astype(np.int16, copy=False)  # Load sampled integer labels exactly as the original cache path does
    feature_names = json.loads(cache_features.read_text(encoding="utf-8"))  # Load ordered readable feature names from JSON
    return features, labels, feature_names  # Return the complete reusable sample cache


def build_and_cache_sample(args: argparse.Namespace, csv_files: Sequence[Path], cache_x: Path, cache_y: Path, cache_features: Path, cache_report: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Inspect source schemas, build the sampled real-data matrix, and persist the sample cache.

    :param args: Validated command-line namespace controlling schema and sampling behavior.
    :param csv_files: Ordered source CSV files discovered recursively.
    :param cache_x: Destination .npy path for sampled features.
    :param cache_y: Destination .npy path for sampled labels.
    :param cache_features: Destination JSON path for readable feature names.
    :param cache_report: Destination JSON path for the sampling audit report.
    :return: Sampled feature matrix, label vector, and ordered readable feature names.
    """

    schemas, feature_keys, display_names = inspect_schemas(
        csv_files,
        include_identifiers=args.include_identifiers,
        keep_inbound=args.keep_inbound,
    )  # Inspect exact headers and compute common usable features across every source CSV
    feature_names = [display_names[key] for key in feature_keys]  # Convert normalized feature keys to stable readable names
    print(f"[DATA] Common usable features across every CSV: {len(feature_keys)}")  # Report the final common feature count
    features, labels, sampling_report = build_memory_safe_sample(
        schemas=schemas,
        feature_keys=feature_keys,
        chunksize=args.chunksize,
        rows_per_file_per_class=args.rows_per_file_per_class,
        global_class_cap=args.global_class_cap,
        prebalance_downsample=args.prebalance_downsample,
        seed=args.data_seed,
        root=args.data_dir,
    )  # Stream and sample the complete raw dataset without modifying source files
    print(
        f"[DATA] Final sampled real-data matrix: rows={len(labels):,}, features={features.shape[1]}, "
        f"RAM={features.nbytes / 2**30:.2f} GiB"
    )  # Report final sampled data shape and in-memory feature size
    save_base_cache_with_eta(features, labels, cache_x, cache_y)  # Persist sampled feature and label arrays with ETA reporting
    cache_features.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")  # Persist ordered readable feature names
    cache_report.write_text(json.dumps(sampling_report, indent=2), encoding="utf-8")  # Persist the sampling audit report
    print(f"[DATA] Base sample cache saved under {args.output_dir}; future runs can use --reuse-sample-cache.")  # Report successful reusable cache persistence
    return features, labels, feature_names  # Return the newly built sampled dataset


def load_or_build_sample(args: argparse.Namespace, csv_files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Reuse the sampled dataset when explicitly requested and complete, otherwise rebuild it.

    :param args: Validated command-line namespace controlling cache reuse and sampling.
    :param csv_files: Ordered source CSV files used when a cache rebuild is required.
    :return: Sampled feature matrix, label vector, and ordered readable feature names.
    """

    cache_x = args.output_dir / "sampled_X.npy"  # Resolve the original sampled feature-cache filename
    cache_y = args.output_dir / "sampled_y.npy"  # Resolve the original sampled label-cache filename
    cache_features = args.output_dir / "feature_names.json"  # Resolve the original sampled feature-name cache filename
    cache_report = args.output_dir / "sampling_report.json"  # Resolve the original sampling-report filename
    if args.reuse_sample_cache and cache_x.exists() and cache_y.exists() and cache_features.exists():  # Verify if explicit reuse was requested and the required cache files exist
        return load_cached_sample(cache_x, cache_y, cache_features)  # Reuse the existing sampled dataset without rescanning source contents
    return build_and_cache_sample(args, csv_files, cache_x, cache_y, cache_features, cache_report)  # Build and persist a fresh sampled dataset
