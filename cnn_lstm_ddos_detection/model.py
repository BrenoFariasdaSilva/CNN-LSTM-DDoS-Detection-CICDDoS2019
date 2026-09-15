"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 CNN-LSTM MODEL DEFINITION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Defines the TensorFlow CNN-LSTM reconstruction used for CICDDoS2019 multiclass
    classification and the memory-aware tf.data input pipeline used during training.

    Key features include:
        - Implements CNN feature extraction followed by parallel Dense and LSTM branches.
        - Preserves the Metal-safe MatMul -> AddV2 -> ReLU dense-layer implementation.
        - Produces 12-class softmax probabilities trained with categorical cross-entropy.

Usage:
    1. Build the model with build_model() using a validated Config.
    2. Build batched TensorFlow datasets with make_tf_dataset().
    3. Supply the custom layer when deserializing saved Keras checkpoints.

Outputs:
    - Compiled tf.keras.Model instances and tf.data.Dataset input pipelines.

TODOs:
    - None identified.

Dependencies:
    - numpy.
    - tensorflow.
    - cnn_lstm_ddos_detection.config.

Assumptions & Notes:
    - The feature axis is treated as the CNN/LSTM sequence axis, preserving the original
      reconstruction of the publication's undocumented sequence construction.
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np
import tensorflow as tf

from .config import Config


class MetalSafeDenseReLU(tf.keras.layers.Layer):
    """Implement Dense + ReLU as MatMul -> AddV2 -> ReLU for Metal compatibility."""

    units: int
    kernel: tf.Variable
    bias: tf.Variable

