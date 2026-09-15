"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 PROGRESS AND RESOURCE REPORTING
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Implements duration formatting, ETA calculations, byte-scan progress reporting,
    and epoch resource logging used by the CNN-LSTM CICDDoS2019 reproduction pipeline.

    Key features include:
        - Formats elapsed and estimated durations consistently across stages.
        - Reports progress for long raw-dataset scans using consumed bytes.
        - Reports TensorFlow epoch ETA and process/system memory usage.

Usage:
    1. Use format_duration() and eta_from_progress() for stage progress messages.
    2. Create byte progress reporters with create_byte_progress().
    3. Create Keras resource callbacks with create_epoch_resource_logger().

Outputs:
    - Progress and resource messages written to standard output.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - psutil.
    - tensorflow.

Assumptions & Notes:
    - Byte-based scan progress is an estimate because pandas may buffer input data.
================================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import tensorflow as tf


def format_duration(seconds: Optional[float]) -> str:
    """
    Format a duration in a compact human-readable representation.

    :param seconds: Duration in seconds or None when the duration is unavailable.
    :return: Formatted duration string.
    """

    if seconds is None or not np.isfinite(seconds) or seconds < 0:  # Verify if the duration cannot be represented safely
        return "unavailable"  # Return the original unavailable marker for invalid durations
    rounded_seconds = int(round(seconds))  # Round seconds to match the original display behavior
    days, remainder = divmod(rounded_seconds, 86400)  # Split complete days from the remaining seconds
    hours, remainder = divmod(remainder, 3600)  # Split complete hours from the remaining seconds
    minutes, secs = divmod(remainder, 60)  # Split complete minutes from the remaining seconds
    if days:  # Verify if the duration spans at least one complete day
        return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"  # Return the day-level duration format
    if hours:  # Verify if the duration spans at least one complete hour
        return f"{hours}h {minutes:02d}m {secs:02d}s"  # Return the hour-level duration format
    if minutes:  # Verify if the duration spans at least one complete minute
        return f"{minutes}m {secs:02d}s"  # Return the minute-level duration format
    return f"{secs}s"  # Return a seconds-only duration for short intervals
