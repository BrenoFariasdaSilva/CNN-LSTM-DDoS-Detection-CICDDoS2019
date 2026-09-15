"""
================================================================================
DUAL-CHANNEL CONSOLE AND FILE LOGGER
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-15
Description :
    Provides a stream-compatible logger that mirrors program output to the active
    terminal and to an ANSI-clean UTF-8 log file. It is used by the project entry
    point to preserve live console output while recording complete experiment logs.

    Key features include:
        - Mirrors stdout and stderr messages to the console and a persistent log file.
        - Removes ANSI terminal escape sequences from the file copy only.
        - Flushes output immediately and supports common stream inspection methods.

Usage:
    1. Import Logger from the repository-root Logger.py module.
    2. Create an instance with Logger.create(path, clean=True).
    3. Assign the instance to sys.stdout and sys.stderr.

Outputs:
    - Writes cleaned console output to the configured log file.
    - Preserves the same output on the original console stream.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - The logger does not close or otherwise manage the original terminal stream.
    - File writes are serialized with a re-entrant lock for safe callback output.
================================================================================
"""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


ANSI_ESCAPE_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Functions Definitions:

@dataclass
class Logger:
    """Mirror text-stream output to a terminal and an ANSI-clean log file."""

    logfile_path: Path
    logfile: TextIO
    terminal_stream: TextIO
    terminal_is_tty: bool
    lock: Any