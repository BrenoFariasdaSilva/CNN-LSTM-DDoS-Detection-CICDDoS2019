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
