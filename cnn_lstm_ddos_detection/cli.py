"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 COMMAND-LINE INTERFACE
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines and validates the command-line interface for the CNN-LSTM CICDDoS2019 reproduction.
    Existing option names/defaults are preserved while zero now explicitly enables uncapped per-file sampling.

    Key features include:
        - Parses dataset, preprocessing, model, training, and runtime options.
        - Validates argument ranges before the workflow starts.
        - Supports 0 as the retain-all value for per-file and global sampling caps.
        - Resolves relative output directories against the project root.

Usage:
    1. Call parse_args() from the top-level orchestrator.
    2. Call validate_args() before constructing Config.
    3. Resolve the output path with resolve_output_dir().

Outputs:
    - Parsed and validated command-line configuration values.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.
    - cnn_lstm_ddos_detection.constants.

Assumptions & Notes:
    - Relative output paths resolve from the directory containing main.py, matching the
      original implementation's output-location behavior.
    - Positive sampling caps preserve the original bounded behavior; zero enables full-source execution.
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .constants import PAPER_TARGET_ACCURACY, PROJECT_ROOT


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the reproduction pipeline.

    :param argv: Optional explicit argument sequence; uses process arguments when None.
    :return: Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description="CNN-LSTM DDoS Detection on CICDDoS2019: memory-safe independent reproduction of Rajput & Upadhyay (2024) with training-only SMOTE."
    )  # Create the command-line parser with the original description
    parser.add_argument("--data-dir", type=Path, required=True, help="READ-ONLY root containing BOTH CICDDoS2019 day folders and their CSVs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CNN-LSTM-DDoS-Detection-100k"),
        help="Generated output. Relative paths are resolved against the directory containing main.py.",
    )
    parser.add_argument("--chunksize", type=int, default=50_000, help="Rows read from disk at once.")
    parser.add_argument(
        "--rows-per-file-per-class",
        type=int,
        default=100_000,
        help="Maximum uniform sample retained from each CSV for each target class. 0=no per-file cap (retain all target rows).",
    )
    parser.add_argument(
        "--global-class-cap",
        type=int,
        default=100_000,
        help="Maximum real rows retained globally per class before the split. 0=no global cap.",
    )
    parser.add_argument(
        "--prebalance-downsample",
        action="store_true",
        help="Diagnostic only: downsample before split. OFF by default so training-only SMOTE handles imbalance.",
    )
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--reuse-sample-cache", action="store_true", help="Reuse sampled_X.npy/sampled_y.npy if present.")
    parser.add_argument("--runs", type=int, default=1, help="Paper does not state run count; 5 is recommended for robustness.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30, help="Not reported by paper.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size; configure for the available accelerator memory.")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping because paper does not report it.")
    parser.add_argument("--smote-k-neighbors", type=int, default=5, help="Classic SMOTE neighbor count; not reported by paper.")
    parser.add_argument("--smote-generation-chunk", type=int, default=50_000, help="Synthetic rows generated per in-memory SMOTE batch.")
    parser.add_argument("--smote-neighbor-query-chunk", type=int, default=512, help="Rows queried per exact SMOTE k-NN batch; controls memory and ETA granularity.")
    parser.add_argument("--conv-filters-1", type=int, default=64)
    parser.add_argument("--conv-filters-2", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--post-dense-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0, help="Paper does not report dropout; default 0.")
    parser.add_argument("--keep-inbound", action="store_true")
    parser.add_argument("--include-identifiers", action="store_true", help="Not recommended; may create leakage.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow execution when no TensorFlow GPU is detected.")
    parser.add_argument("--deterministic-ops", action="store_true", help="Optional; may reduce accelerator compatibility/performance.")
    parser.add_argument("--mixed-precision", action="store_true", help="Optional; float32 is the reproduction default.")
    parser.add_argument(
        "--no-save-derived-data",
        action="store_true",
        help="Do not persist standardized/SMOTE train and standardized validation/test copies. Default is to save them.",
    )
    parser.add_argument("--target-accuracy", type=float, default=PAPER_TARGET_ACCURACY)
    return parser.parse_args(argv)  # Return parsed values without starting any pipeline stage
