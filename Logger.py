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

    @classmethod


    def create(cls: type[Logger], logfile_path: str | Path, clean: bool = False) -> Logger:
        """
        Create a logger connected to the original standard-output terminal.

        :param cls: Logger class used to construct the instance.
        :param logfile_path: Destination path for the persistent log file.
        :param clean: Whether to truncate an existing log file before writing.
        :return: Configured Logger instance.
        """

        resolved_path = Path(logfile_path).expanduser().resolve()  # Resolve the requested log destination
        resolved_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the logs directory exists before opening the file
        mode = "w" if clean else "a"  # Select truncation or append behavior
        logfile = resolved_path.open(mode, encoding="utf-8", buffering=1)  # Open a line-buffered UTF-8 log file
        terminal_stream = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout  # Preserve the original console stream
        terminal_is_tty = bool(terminal_stream.isatty())  # Detect whether ANSI output is appropriate for the console
        return cls(resolved_path, logfile, terminal_stream, terminal_is_tty, threading.RLock())  # Build the stream-compatible logger


    def write(self: Logger, message: str) -> int:
        """
        Write one stream fragment to the terminal and cleaned log file.

        :param self: Logger instance receiving the stream fragment.
        :param message: Text fragment supplied by print or another stream writer.
        :return: Number of characters accepted from the original message.
        """

        output = str(message)  # Normalize the incoming stream fragment without changing newline placement
        clean_output = ANSI_ESCAPE_REGEX.sub("", output)  # Remove terminal escape sequences from the persistent copy
        with self.lock:  # Serialize terminal and file writes from callbacks or worker threads
            try:  # Protect the experiment from non-critical log-file failures
                self.logfile.write(clean_output)  # Persist the ANSI-clean stream fragment
                self.logfile.flush()  # Keep the log file current during long-running experiments
            except Exception:  # Ignore logging failures so they do not terminate model execution
                pass  # Preserve the caller's original execution behavior
            try:  # Protect the experiment from non-critical terminal failures
                terminal_output = output if self.terminal_is_tty else clean_output  # Preserve ANSI only for interactive terminals
                self.terminal_stream.write(terminal_output)  # Mirror the same stream fragment to the console
                self.terminal_stream.flush()  # Display progress messages immediately
            except Exception:  # Ignore terminal failures in detached or closing processes
                pass  # Preserve the caller's original execution behavior
        return len(output)  # Report the accepted character count expected by text streams


    def flush(self: Logger) -> None:
        """
        Flush both the persistent log and original terminal streams.

        :param self: Logger instance whose streams must be flushed.
        :return: None.
        """

        with self.lock:  # Prevent flush operations from interleaving with active writes
            try:  # Protect the experiment from non-critical log flush failures
                self.logfile.flush()  # Force buffered log content to disk
            except Exception:  # Ignore flush failures during interpreter shutdown
                pass  # Preserve normal shutdown behavior
            try:  # Protect the experiment from non-critical terminal flush failures
                self.terminal_stream.flush()  # Force buffered console content to display
            except Exception:  # Ignore flush failures in detached or closing terminals
                pass  # Preserve normal shutdown behavior


    def close(self: Logger) -> None:
        """
        Flush and close only the persistent log file.

        :param self: Logger instance whose file must be closed.
        :return: None.
        """

        with self.lock:  # Prevent the file from closing during an active write
            self.flush()  # Persist any remaining output before closing
            try:  # Protect interpreter shutdown from repeated close calls
                self.logfile.close()  # Release the persistent log-file handle
            except Exception:  # Ignore close failures during interpreter shutdown
                pass  # Preserve normal shutdown behavior


    def isatty(self: Logger) -> bool:
        """
        Report whether the original terminal stream is interactive.

        :param self: Logger instance queried by terminal-aware libraries.
        :return: True when the original terminal stream is a TTY.
        """

        return self.terminal_is_tty  # Forward the original terminal's interactivity state
