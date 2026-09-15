"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 REPRODUCTION PACKAGE
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Marks the modular CNN-LSTM CICDDoS2019 reproduction implementation as a Python package.
    Pipeline stages are implemented in focused modules and orchestrated by main.py.

    Key features include:
        - Exposes a package boundary for the reproduction implementation.
        - Keeps the top-level main.py focused on orchestration.
        - Supports explicit imports between independent pipeline stages.

Usage:
    1. Execute the project through the top-level main.py file.
    2. Import individual modules only when testing or reusing a pipeline stage.
    3. Keep raw CICDDoS2019 data outside this package directory.

Outputs:
    - None directly produced.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - The package intentionally does not execute the experiment during import.
================================================================================
"""

from __future__ import annotations
