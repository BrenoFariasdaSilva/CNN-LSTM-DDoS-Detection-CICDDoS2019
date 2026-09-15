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


def eta_from_progress(done: float, total: float, elapsed: float) -> Optional[float]:
    """
    Estimate remaining seconds from completed work and elapsed time.

    :param done: Amount of work completed so far.
    :param total: Total amount of work expected.
    :param elapsed: Elapsed time in seconds.
    :return: Estimated remaining seconds or None when an estimate is unavailable.
    """

    if done <= 0 or total <= 0 or elapsed <= 0:  # Verify if enough positive progress information exists
        return None  # Return no ETA until a meaningful rate can be calculated
    rate = done / elapsed  # Calculate the observed processing rate
    if rate <= 0:  # Verify if the calculated processing rate is usable
        return None  # Return no ETA for a non-positive processing rate
    return max(0.0, (total - done) / rate)  # Estimate and clamp the remaining duration to zero or greater


@dataclass
class ByteProgress:
    """Store state for byte-based progress reporting during source-data scans."""

    total_bytes: int
    label: str
    start: float
    last_print: float
    min_interval: float

    def report(self: "ByteProgress", done_bytes: int, detail: str = "", force: bool = False) -> None:
        """
        Report byte-scan progress when the minimum reporting interval has elapsed.

        :param self: Current ByteProgress reporter instance.
        :param done_bytes: Number of source bytes consumed so far.
        :param detail: Optional stage detail appended to the progress line.
        :param force: Whether to report even when the minimum interval has not elapsed.
        :return: None.
        """

        now = time.time()  # Capture one timestamp for all calculations in this report
        if not force and now - self.last_print < self.min_interval:  # Verify if a non-forced report is being requested too soon
            return  # Skip frequent progress output to preserve the original reporting cadence
        done = min(max(int(done_bytes), 0), self.total_bytes)  # Clamp consumed bytes to the valid scan range
        elapsed = now - self.start  # Calculate elapsed scan time
        eta = eta_from_progress(done, self.total_bytes, elapsed)  # Estimate remaining scan time
        pct = 100.0 * done / self.total_bytes  # Calculate completion percentage
        rate_mib = done / max(elapsed, 1e-9) / (1024 ** 2)  # Calculate the observed scan throughput in MiB/s
        suffix = f" | {detail}" if detail else ""  # Preserve the optional detail suffix format
        print(
            f"[ETA][{self.label}] {pct:6.2f}% | "
            f"{done / 2**30:.2f}/{self.total_bytes / 2**30:.2f} GiB | "
            f"{rate_mib:.1f} MiB/s | elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta)}{suffix}"
        )  # Emit the original byte-progress fields
        self.last_print = now  # Record the report timestamp for interval throttling
