"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 REPRODUCTION CONSTANTS
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines the fixed class mapping, label aliases, feature-exclusion keys, and
    paper target accuracy used throughout the CNN-LSTM CICDDoS2019 reproduction pipeline.

    Key features include:
        - Defines the inferred 12-class CICDDoS2019 target mapping.
        - Defines raw-label aliases used during dataset ingestion.
        - Defines default feature keys excluded from model inputs.

Usage:
    1. Import the required constants from other cnn_lstm_ddos_detection package modules.
    2. Do not modify the raw dataset through this module; it contains constants only.
    3. Keep class ordering unchanged when reproducing existing experiment outputs.

Outputs:
    - None directly produced.

TODOs:
    - None identified.

Dependencies:
    - Python standard library.

Assumptions & Notes:
    - The publication states 12 classes but does not enumerate them; the mapping here
      preserves the reconstruction already implemented by the original main.py.
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

PAPER_TARGET_ACCURACY = 0.9976
