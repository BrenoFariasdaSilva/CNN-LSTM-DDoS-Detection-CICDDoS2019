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


def configure_numeric_policy(mixed_precision: bool) -> None:
    """
    Configure the TensorFlow global floating-point policy.

    :param mixed_precision: Whether mixed_float16 should be enabled.
    :return: None.
    """

    if mixed_precision:  # Verify if optional mixed precision was requested explicitly
        tf.keras.mixed_precision.set_global_policy("mixed_float16")  # Enable the original optional mixed-precision policy
        print("[SYSTEM] Mixed precision enabled: mixed_float16")  # Report the active numeric policy
    else:  # Handle the reproducibility-oriented default policy
        tf.keras.mixed_precision.set_global_policy("float32")  # Preserve float32 as the default numeric policy
        print("[SYSTEM] Numeric policy: float32")  # Report the active default numeric policy


def configure_accelerator(allow_cpu: bool, mixed_precision: bool) -> str:
    """
    Configure and verify the TensorFlow execution device used by the experiment.

    :param allow_cpu: Whether CPU execution is allowed when no GPU is detected.
    :param mixed_precision: Whether mixed_float16 should be enabled globally.
    :return: TensorFlow device name selected for experiment execution.
    """

    print("[SYSTEM] platform:", platform.platform())  # Report the operating-system platform
    print("[SYSTEM] machine:", platform.machine())  # Report the processor architecture
    print("[SYSTEM] Python:", sys.version.split()[0])  # Report the active Python version
    print("[SYSTEM] TensorFlow:", tf.__version__)  # Report the active TensorFlow version
    print("[SYSTEM] tensorflow-metal:", package_version("tensorflow-metal"))  # Report the installed Metal plugin version
    tf.config.set_soft_device_placement(True)  # Preserve TensorFlow soft placement behavior
    gpus = tf.config.list_physical_devices("GPU")  # Discover physical GPU devices visible to TensorFlow
    print("[SYSTEM] TensorFlow GPU devices:", gpus)  # Report discovered GPU devices
    if not gpus:  # Verify if TensorFlow found no GPU devices
        if not allow_cpu:  # Verify if the user did not explicitly permit CPU execution
            raise RuntimeError(
                "No TensorFlow GPU was detected. On Apple Silicon verify tensorflow-metal; "
                "on Linux verify the NVIDIA driver and TensorFlow CUDA dependencies. Install "
                "requirements.txt in a clean Python 3.11/3.12 venv, or use --allow-cpu only "
                "if you intentionally want CPU training."
            )  # Reject missing GPU acceleration unless CPU fallback was explicitly allowed
        print("[SYSTEM] WARNING: running on CPU because --allow-cpu was supplied.")  # Report explicit CPU fallback
        device = "/CPU:0"  # Select the CPU device for the complete run
    else:  # Handle a TensorFlow-visible GPU on either supported platform
        device = "/GPU:0"  # Select the first TensorFlow GPU device
        run_gpu_smoke_test(device)  # Verify that a real operation is placed on the GPU
    configure_numeric_policy(mixed_precision)  # Configure the requested global numeric policy
    return device  # Return the verified execution device


def set_seeds(seed: int, deterministic_ops: bool) -> None:
    """
    Apply one experiment seed across Python, NumPy, and TensorFlow.

    :param seed: Integer random seed for the current experiment run.
    :param deterministic_ops: Whether TensorFlow deterministic operations should be requested.
    :return: None.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)  # Set the Python hash seed value for reproducibility metadata
    random.seed(seed)  # Seed Python's standard random module
    np.random.seed(seed)  # Seed NumPy's legacy global random generator
    tf.keras.utils.set_random_seed(seed)  # Seed TensorFlow and Keras random generators
    if deterministic_ops:  # Verify if deterministic TensorFlow operations were explicitly requested
        try:  # Request deterministic operations where the installed TensorFlow build supports them
            tf.config.experimental.enable_op_determinism()  # Enable deterministic TensorFlow operation selection
            print("[SYSTEM] TensorFlow deterministic ops enabled.")  # Report successful deterministic-mode activation
        except Exception as exc:  # Preserve the original non-fatal behavior when deterministic mode is unsupported
            print(f"[SYSTEM] Could not enable deterministic ops: {exc}")  # Report the unsupported deterministic-mode request


def environment_info(device: str) -> Dict[str, object]:
    """
    Collect runtime environment metadata for experiment auditability.

    :param device: TensorFlow device selected for experiment execution.
    :return: Dictionary containing runtime, package, device, and memory metadata.
    """

    virtual_memory = psutil.virtual_memory()  # Read total system memory for the environment record
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "tensorflow_metal": package_version("tensorflow-metal"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device_selected": device,
        "physical_gpus": [str(device_info) for device_info in tf.config.list_physical_devices("GPU")],
        "system_ram_gib": virtual_memory.total / (1024 ** 3),
    }  # Return metadata using the original environment-information keys
