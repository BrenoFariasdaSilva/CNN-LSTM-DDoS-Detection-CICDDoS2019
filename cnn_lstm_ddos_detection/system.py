"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 RUNTIME AND ACCELERATOR SETUP
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Configures TensorFlow runtime behavior, GPU acceleration, random seeds, deterministic
    execution options, and environment metadata for macOS Apple Silicon and Linux runs.

    Key features include:
        - Verifies TensorFlow-visible GPU availability with a real smoke test.
        - Configures float32 or optional mixed_float16 numeric policy.
        - Applies Python, NumPy, and TensorFlow random seeds consistently.

Usage:
    1. Call configure_accelerator() once before dataset processing and model training.
    2. Call set_seeds() for each run using its configured seed.
    3. Persist environment_info() output for experiment auditability.

Outputs:
    - Runtime diagnostics written to standard output.
    - Environment metadata returned to the caller for persistence.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - pandas.
    - psutil.
    - tensorflow.
    - tensorflow-metal on Apple Silicon; TensorFlow CUDA dependencies on supported Linux NVIDIA hosts.

Assumptions & Notes:
    - CPU execution is rejected unless --allow-cpu is explicitly supplied.
================================================================================
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # Keep TensorFlow startup output at the original verbosity

try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError(
        "TensorFlow could not be imported. Use Python 3.11/3.12 and install this project's "
        "platform-aware requirements.txt (tensorflow-metal on Apple Silicon or TensorFlow "
        "CUDA dependencies on supported Linux NVIDIA hosts). Original error:\n"
        f"{exc}"
    ) from exc


def package_version(name: str) -> Optional[str]:
    """
    Return an installed package version when package metadata is available.

    :param name: Distribution package name to query.
    :return: Installed version string or None when the package is not installed.
    """

    try:  # Query distribution metadata without importing the requested package
        return importlib.metadata.version(name)  # Return the installed distribution version
    except importlib.metadata.PackageNotFoundError:  # Handle packages that are not installed in the environment
        return None  # Preserve the original missing-package representation


def run_gpu_smoke_test(device: str) -> None:
    """
    Execute and verify a real TensorFlow matrix multiplication on the selected GPU.

    :param device: TensorFlow device name selected for GPU execution.
    :return: None.
    """

    with tf.device(device):  # Force the smoke-test operations onto the selected TensorFlow device
        left = tf.random.uniform((256, 256), dtype=tf.float32)  # Generate the first smoke-test matrix
        right = tf.random.uniform((256, 256), dtype=tf.float32)  # Generate the second smoke-test matrix
        product = tf.linalg.matmul(left, right)  # Execute a real matrix multiplication through TensorFlow
        checksum = float(tf.reduce_sum(product).numpy())  # Materialize the result so device execution actually occurs
    print(f"[SYSTEM] GPU smoke test device={product.device} checksum={checksum:.3f}")  # Report observed TensorFlow placement
    if "GPU" not in product.device.upper():  # Verify if TensorFlow placed the smoke-test result on a GPU
        raise RuntimeError(
            f"A GPU was listed but the smoke-test operation was placed on {product.device}. "
            "Refusing to claim GPU acceleration."
        )  # Reject a falsely advertised GPU configuration
