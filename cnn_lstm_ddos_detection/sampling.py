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


def update_class_reservoirs(features: np.ndarray, labels: np.ndarray, reservoirs_x: Dict[str, Optional[np.ndarray]], reservoirs_p: Dict[str, Optional[np.ndarray]], rows_per_class: int, rng: np.random.Generator) -> None:
    """
    Update every class reservoir from one numeric streamed chunk.

    :param features: Numeric feature matrix for target rows in the current chunk.
    :param labels: Canonical target labels aligned with the feature matrix.
    :param reservoirs_x: Mutable per-class retained feature reservoirs.
    :param reservoirs_p: Mutable per-class retained priority reservoirs.
    :param rows_per_class: Maximum retained rows per class for this source file.
    :param rng: NumPy generator dedicated to this source file.
    :return: None.
    """

    for class_name in PAPER_12_CLASSES:  # Process classes in the fixed original class order
        class_indices = np.flatnonzero(labels == class_name)  # Locate current-chunk rows for this class
        if class_indices.size == 0:  # Verify if the class is absent from the current chunk
            continue  # Skip RNG consumption for absent classes exactly as before
        new_features = features[class_indices]  # Select candidate feature rows for the class
        new_priorities = rng.random(class_indices.size, dtype=np.float64)  # Draw uniform priorities for every candidate row
        if len(new_priorities) > rows_per_class:  # Verify if this chunk alone exceeds the per-file class cap
            keep = np.argpartition(new_priorities, rows_per_class - 1)[:rows_per_class]  # Keep the smallest candidate priorities
            new_features = new_features[keep]  # Retain feature rows aligned with selected priorities
            new_priorities = new_priorities[keep]  # Retain the selected priority values
        retained_features, retained_priorities = priority_reservoir_merge(
            reservoirs_x[class_name],
            reservoirs_p[class_name],
            new_features,
            new_priorities,
            rows_per_class,
        )  # Merge candidates into the bounded class reservoir
        reservoirs_x[class_name] = retained_features  # Store the updated feature reservoir
        reservoirs_p[class_name] = retained_priorities  # Store the updated priority reservoir


def collect_unlimited_class_pieces(features: np.ndarray, labels: np.ndarray, class_pieces: Dict[str, List[np.ndarray]]) -> None:
    """
    Append every target row from one streamed chunk to per-class retain-all pieces.

    :param features: Numeric feature matrix for target rows in the current chunk.
    :param labels: Canonical target labels aligned with the feature matrix.
    :param class_pieces: Mutable per-class lists collecting uncapped feature chunks.
    :return: None.
    """

    for class_name in PAPER_12_CLASSES:  # Process classes in the fixed target-class order
        class_indices = np.flatnonzero(labels == class_name)  # Locate all current-chunk rows for this class
        if class_indices.size == 0:  # Verify if the class is absent from the current chunk
            continue  # Skip empty class pieces without allocating an array
        class_pieces[class_name].append(features[class_indices])  # Retain every target row for this class without sampling


def estimate_consumed_bytes(raw_handle: object, chunk_index: int, chunksize: int, file_size: int) -> int:
    """
    Estimate source bytes consumed by a pandas CSV chunk reader.

    :param raw_handle: Binary source-file handle used by pandas.
    :param chunk_index: One-based chunk index currently completed.
    :param chunksize: Configured pandas chunk row count.
    :param file_size: Total size of the current source file in bytes.
    :return: Estimated consumed bytes clamped to the source file size.
    """

    try:  # Prefer the underlying file handle position when pandas exposes it reliably
        return min(int(raw_handle.tell()), file_size)  # Return consumed bytes reported by the binary file handle
    except Exception:  # Preserve the original fallback for file-position reporting failures
        return min(chunk_index * chunksize * 256, file_size)  # Estimate bytes from rows using the original fallback multiplier


def stream_sample_one_file(schema: FileSchema, feature_keys: Sequence[str], chunksize: int, rows_per_class: int, seed: int, scan_progress: ByteProgress, bytes_before_file: int) -> Tuple[Dict[str, np.ndarray], Counter, Counter]:
    """
    Stream one CSV and retain the configured rows per target class.

    :param schema: Exact source CSV schema metadata.
    :param feature_keys: Ordered common normalized feature keys.
    :param chunksize: Number of CSV rows read per pandas chunk.
    :param rows_per_class: Maximum retained real rows for each class in this file; zero retains all.
    :param seed: Random seed dedicated to this source file.
    :param scan_progress: Shared byte-progress reporter across all source files.
    :param bytes_before_file: Total source bytes belonging to previously processed files.
    :return: Per-class retained arrays plus observed target and omitted-class counters.
    """

    rng = np.random.default_rng(seed)  # Create the same independent per-file generator used by bounded sampling
    reservoirs_x: Dict[str, Optional[np.ndarray]] = {class_name: None for class_name in PAPER_12_CLASSES}  # Initialize bounded feature reservoirs
    reservoirs_p: Dict[str, Optional[np.ndarray]] = {class_name: None for class_name in PAPER_12_CLASSES}  # Initialize bounded priority reservoirs
    unlimited_pieces: Dict[str, List[np.ndarray]] = {class_name: [] for class_name in PAPER_12_CLASSES}  # Initialize uncapped class chunk lists
    unlimited_retained_total = 0  # Track uncapped retained rows without rescanning accumulated chunk lists
    observed = Counter()  # Count recognized target-class rows seen in the complete file
    omitted = Counter()  # Count recognized non-target rows seen in the complete file
    actual_feature_columns = [schema.columns_by_key[key] for key in feature_keys]  # Resolve exact source headers once per file
    use_columns = actual_feature_columns + [schema.label_column]  # Read only common features plus the exact label column
    file_size = schema.path.stat().st_size  # Capture source file size for scan progress
    with schema.path.open("rb") as raw_handle:  # Open the raw CSV in binary read-only mode
        reader = pd.read_csv(raw_handle, usecols=use_columns, chunksize=chunksize, low_memory=False)  # Stream source rows without loading the complete file
        for chunk_index, chunk in enumerate(reader, start=1):  # Process every source chunk in source order
            labels = canonicalize_labels(chunk[schema.label_column])  # Normalize raw labels for counting and target selection
            update_observed_counts(labels, observed, omitted)  # Accumulate target and recognized omitted class counts
            features, target_labels = build_numeric_chunk(chunk, schema, feature_keys, labels)  # Convert target rows to the common numeric schema
            if len(target_labels) > 0:  # Verify if the chunk contains at least one target row
                if rows_per_class == 0:  # Verify if explicit retain-all mode is enabled for Linux/full-data execution
                    collect_unlimited_class_pieces(features, target_labels, unlimited_pieces)  # Preserve every target row without consuming sampling RNG state
                    unlimited_retained_total += len(target_labels)  # Advance uncapped retained-row progress by this chunk
                else:  # Handle the existing bounded priority-reservoir sampling path
                    update_class_reservoirs(features, target_labels, reservoirs_x, reservoirs_p, rows_per_class, rng)  # Update bounded per-class sampling state
            consumed = estimate_consumed_bytes(raw_handle, chunk_index, chunksize, file_size)  # Estimate source bytes consumed so far
            if rows_per_class == 0:  # Verify if retained-row progress must be counted from uncapped chunk lists
                retained_total = unlimited_retained_total  # Reuse the running count of every retained uncapped target row
            else:  # Handle bounded reservoir progress counting
                retained_total = sum(0 if value is None else len(value) for value in reservoirs_x.values())  # Count currently retained bounded rows
            scan_progress.report(
                bytes_before_file + consumed,
                detail=f"{schema.path.name} chunk={chunk_index} retained={retained_total:,}",
            )  # Report global raw-scan progress
            del chunk, features, target_labels  # Release per-chunk objects before continuing the raw scan
            gc.collect()  # Encourage prompt release of temporary chunk memory on constrained systems
    scan_progress.report(bytes_before_file + file_size, detail=f"completed {schema.path.name}", force=True)  # Force a final progress line for this source file
    if rows_per_class == 0:  # Verify if the source file was processed in retain-all mode
        result = {
            class_name: np.concatenate(unlimited_pieces[class_name], axis=0).astype(np.float32, copy=False)
            for class_name in PAPER_12_CLASSES
            if unlimited_pieces[class_name]
        }  # Concatenate each uncapped class once after the complete source file has been streamed
    else:  # Handle the existing bounded reservoir result
        result = {
            class_name: reservoirs_x[class_name]
            for class_name in PAPER_12_CLASSES
            if reservoirs_x[class_name] is not None and len(reservoirs_x[class_name]) > 0
        }  # Remove empty class reservoirs from the returned sample mapping
    return result, observed, omitted  # Return per-file retained rows and audit counters


def collect_file_samples(schemas: Sequence[FileSchema], feature_keys: Sequence[str], chunksize: int, rows_per_file_per_class: int, seed: int, root: Path) -> Tuple[Dict[str, List[np.ndarray]], List[Dict[str, object]], Counter, Counter]:
    """
    Stream all source CSV files and collect bounded or uncapped per-file class rows.

    :param schemas: Ordered source-file schema metadata.
    :param feature_keys: Ordered common normalized feature keys.
    :param chunksize: Number of CSV rows read per pandas chunk.
    :param rows_per_file_per_class: Maximum retained real rows per class from each file; zero retains all.
    :param seed: Dataset sampling seed.
    :param root: Raw dataset root used for readable relative paths.
    :return: Per-class sample pieces, per-file report, total target counts, and total omitted counts.
    """

    per_class_pieces: Dict[str, List[np.ndarray]] = {class_name: [] for class_name in PAPER_12_CLASSES}  # Collect sampled pieces by target class
    file_report: List[Dict[str, object]] = []  # Collect auditable per-file sampling metadata
    all_observed = Counter()  # Accumulate target-class observations across files
    all_omitted = Counter()  # Accumulate recognized omitted-class observations across files
    total_bytes = sum(schema.path.stat().st_size for schema in schemas)  # Calculate total source bytes for global scan progress
    scan_progress = create_byte_progress(total_bytes, "RAW-SCAN")  # Initialize the shared raw-data scan reporter
    bytes_before = 0  # Track completed source bytes before each file
    for index, schema in enumerate(schemas, start=1):  # Stream every source CSV in deterministic schema order
        relative = schema.path.relative_to(root) if schema.path.is_relative_to(root) else schema.path  # Prefer readable paths relative to the dataset root
        print(f"[DATA] file {index}/{len(schemas)}: {relative}")  # Report the source file being processed
        sampled, observed, omitted = stream_sample_one_file(
            schema=schema,
            feature_keys=feature_keys,
            chunksize=chunksize,
            rows_per_class=rows_per_file_per_class,
            seed=seed + index * 1009,
            scan_progress=scan_progress,
            bytes_before_file=bytes_before,
        )  # Stream and sample the current source file using the original seed derivation
        bytes_before += schema.path.stat().st_size  # Advance completed-source byte accounting
        all_observed.update(observed)  # Merge this file's target-class counts
        all_omitted.update(omitted)  # Merge this file's recognized omitted-class counts
        retained = {class_name: int(len(sampled[class_name])) if class_name in sampled else 0 for class_name in PAPER_12_CLASSES}  # Record per-class retained counts
        for class_name, array in sampled.items():  # Preserve each sampled class array for global capping
            per_class_pieces[class_name].append(array)  # Append this file's retained class sample
        file_report.append(
            {
                "file": str(relative),
                "observed_target_counts": dict(observed),
                "retained_counts": retained,
                "omitted_counts": dict(omitted),
            }
        )  # Preserve the original per-file report structure
        print(f"      retained from file: {retained}")  # Report retained rows from the current source file
        gc.collect()  # Encourage release of temporary per-file objects
    return per_class_pieces, file_report, all_observed, all_omitted  # Return globally accumulated sampling inputs and audit data


def build_capped_class_arrays(per_class_pieces: Dict[str, List[np.ndarray]], global_class_cap: int, prebalance_downsample: bool, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Optional[int]]:
    """
    Concatenate sampled pieces, apply the global class cap, and optionally pre-balance.

    :param per_class_pieces: Sampled feature-array pieces grouped by canonical class.
    :param global_class_cap: Maximum real rows retained per class after combining files; zero disables the cap.
    :param prebalance_downsample: Whether diagnostic equal-size pre-balancing is enabled.
    :param rng: Dataset-level random generator used for capping and pre-balancing.
    :return: Final per-class arrays, counts before global cap, and optional pre-balance size.
    """

    class_arrays: Dict[str, np.ndarray] = {}  # Store concatenated/capped feature arrays for every target class
    counts_before_global_cap: Dict[str, int] = {}  # Record counts after per-file sampling and before global capping
    for class_name in PAPER_12_CLASSES:  # Process target classes in the fixed original order
        pieces = per_class_pieces[class_name]  # Retrieve all sampled file pieces for this class
        if not pieces:  # Verify if the complete corpus yielded no sample for the required class
            raise RuntimeError(
                f"Target class {class_name!r} was not found. Confirm --data-dir points to the full "
                "two-day CICDDoS2019 CSV dataset."
            )  # Reject incomplete datasets that cannot represent all target classes
        array = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)  # Combine this class across every source file
        counts_before_global_cap[class_name] = int(len(array))  # Record the real-row count before global capping
        if global_class_cap > 0 and len(array) > global_class_cap:  # Verify if this class exceeds the optional global cap
            indices = rng.choice(len(array), size=global_class_cap, replace=False)  # Uniformly choose rows without replacement
            array = array[indices]  # Retain only globally capped real rows
        class_arrays[class_name] = array  # Preserve the class array after global capping
    prebalance_count: Optional[int] = None  # Track diagnostic pre-balance size when enabled
    if prebalance_downsample:  # Verify if optional diagnostic pre-balancing was explicitly requested
        prebalance_count = min(len(array) for array in class_arrays.values())  # Determine the smallest retained class size
        print(
            f"[DATA] Diagnostic pre-balance enabled: downsampling all classes to "
            f"{prebalance_count:,} BEFORE split. This is not the default best-result path."
        )  # Report that this non-default diagnostic path is active
        for class_name, array in list(class_arrays.items()):  # Process classes in insertion order while sharing the original RNG
            if len(array) > prebalance_count:  # Verify if this class exceeds the diagnostic target size
                indices = rng.choice(len(array), size=prebalance_count, replace=False)  # Downsample uniformly without replacement
                class_arrays[class_name] = array[indices]  # Store the diagnostically balanced class array
    return class_arrays, counts_before_global_cap, prebalance_count  # Return final class arrays and capping metadata


def combine_class_arrays(class_arrays: Dict[str, np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Combine per-class feature arrays into one shuffled sampled dataset.

    :param class_arrays: Final feature arrays grouped by canonical class.
    :param rng: Dataset-level random generator used for the final row permutation.
    :return: Shuffled feature matrix, integer-label vector, and final class counts.
    """

    feature_parts: List[np.ndarray] = []  # Collect class feature arrays in fixed class order
    label_parts: List[np.ndarray] = []  # Collect aligned integer-label arrays
    final_counts: Dict[str, int] = {}  # Record final sampled real-row counts by class
    for class_id, class_name in enumerate(PAPER_12_CLASSES):  # Assign integer class IDs using the fixed class ordering
        array = class_arrays[class_name]  # Retrieve final sampled rows for this class
        feature_parts.append(array)  # Append class features to the global sampled dataset
        label_parts.append(np.full(len(array), class_id, dtype=np.int16))  # Create aligned integer labels for the class
        final_counts[class_name] = int(len(array))  # Record this class's final real-row count
    features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)  # Concatenate all sampled feature rows
    labels = np.concatenate(label_parts, axis=0)  # Concatenate aligned integer labels
    order = rng.permutation(len(labels))  # Draw the final dataset permutation with the existing dataset RNG state
    return features[order], labels[order], final_counts  # Return the same randomized row order used by the original implementation
