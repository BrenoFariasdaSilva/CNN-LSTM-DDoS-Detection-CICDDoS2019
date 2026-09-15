"""
================================================================================
CNN-LSTM DDOS DETECTION ON CICDDOS2019 ORCHESTRATOR
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Provides the top-level entry point for the modular CICDDoS2019 12-class CNN-LSTM reconstruction of Rajput and Upadhyay (2024). The entry point configures
    dual console/file logging, registers a cross-platform completion notification,
    parses and validates the existing CLI, and delegates the experiment to the modular
    project workflow without changing the model or dataset-processing logic.

    Key features include:
        - Mirrors stdout and stderr to the console and logs/main.log.
        - Registers .assets/Sounds/NotificationSound.wav through atexit.
        - Delegates the complete reproduction workflow to cnn_lstm_ddos_detection.workflow.

Usage:
    1. Configure dataset and output paths through the Makefile variables or CLI options.
    2. Execute make run-mac, make run-linux, or python main.py --data-dir PATH [options].
    3. Review experiment artifacts below --output-dir and runtime output in logs/main.log.

Outputs:
    - Writes the complete runtime console stream to logs/main.log.
    - Writes experiment outputs below the configured --output-dir directory.
    - Plays the bundled completion sound when a supported system audio command is available.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.
    - Logger.py.
    - cnn_lstm_ddos_detection.cli.
    - cnn_lstm_ddos_detection.workflow.

Assumptions & Notes:
    - Raw CICDDoS2019 CSV files remain read-only and are verified by the workflow.
    - Completion-sound failures are non-fatal, including headless Linux audio failures.
    - Relative and absolute output-path behavior remains controlled by the existing CLI.
================================================================================
"""

from __future__ import annotations

import atexit
import datetime
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Logger import Logger


SOUND_COMMANDS: dict[str, tuple[str, ...]] = {
    "Darwin": ("afplay",),
    "Linux": ("aplay", "-q"),
}

# Functions Definitions:


def configure_runtime(project_dir: Path) -> Logger:
    """
    Configure dual-channel logging and register shutdown handlers.

    :param project_dir: Absolute repository root containing main.py.
    :return: Logger assigned to both standard output streams.
    """

    log_file = project_dir / "logs" / "main.log"  # Define the stable runtime log path inside the repository
    sound_file = project_dir / ".assets" / "Sounds" / "NotificationSound.wav"  # Define the bundled completion-sound path
    logger = Logger.create(log_file, clean=True)  # Create a fresh log for this execution
    sys.stdout = logger  # Mirror all standard output to the console and persistent log
    sys.stderr = logger  # Mirror all standard error to the same console and persistent log
    atexit.register(logger.close)  # Close the log after all other registered shutdown output is complete
    atexit.register(play_notification_sound, sound_file)  # Play the completion notification before closing the logger
    print(f"[RUNTIME] Console logging enabled: {log_file}")  # Record the persistent log destination
    print(f"[RUNTIME] Completion sound registered: {sound_file}")  # Record the registered notification asset
    return logger  # Return the configured logger for explicit flushes


def main() -> int:
    """
    Configure runtime services and execute the complete reproduction workflow.

    :return: Zero after successful completion of the reproduction workflow.
    """

    execution_started = time.time()  # Start complete entry-point timing before runtime setup and CLI parsing
    started_at = datetime.datetime.now()  # Capture the human-readable execution start time
    project_dir = Path(__file__).resolve().parent  # Resolve all runtime assets relative to the repository root
    logger = configure_runtime(project_dir)  # Enable persistent logging and register completion playback
    print(f"[RUNTIME] Execution started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")  # Log the start timestamp
    try:  # Ensure finish timing is recorded even when the workflow raises an exception
        from cnn_lstm_ddos_detection.cli import parse_args, validate_args  # Import CLI handling after logging is active
        from cnn_lstm_ddos_detection.workflow import run_workflow  # Import TensorFlow-dependent workflow after logging is active
        args = parse_args()  # Parse the existing project command-line interface
        validate_args(args)  # Normalize paths and reject invalid configuration before dataset processing
        run_workflow(args, execution_started)  # Delegate the full CNN-LSTM CICDDoS2019 pipeline while preserving the original timer origin
        return 0  # Preserve the existing successful process return code
    finally:  # Record final timing before Python runs the registered atexit handlers
        finished_at = datetime.datetime.now()  # Capture the human-readable execution finish time
        elapsed = time.time() - execution_started  # Compute total entry-point execution duration
        print(f"[RUNTIME] Execution finished: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")  # Log the finish timestamp
        print(f"[RUNTIME] Total execution time: {format_execution_duration(elapsed)}")  # Log the human-readable total duration
        logger.flush()  # Persist the final runtime messages before interpreter shutdown


if __name__ == "__main__":  # Verify if this file is being executed as the program entry point
    raise SystemExit(main())  # Preserve the existing process exit behavior
