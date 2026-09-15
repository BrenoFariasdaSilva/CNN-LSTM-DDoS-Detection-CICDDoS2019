"""
================================================================================
CNN-LSTM DDOS DETECTION CICDDOS2019 SINGLE-RUN EXPERIMENT EXECUTION
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-09-14
Description :
    Executes one complete CNN-LSTM CICDDoS2019 reproduction run from stratified splitting through
    preprocessing, training-only SMOTE, CNN-LSTM training, held-out evaluation, and output.

    Key features include:
        - Keeps preprocessing, model preparation, training, evaluation, and persistence separate.
        - Uses one-hot 12-class targets with validation during categorical-cross-entropy training.
        - Persists checkpoints, metrics, predictions, confusion matrices, and fitted transformers.

Usage:
    1. Supply a validated Config, selected TensorFlow device, sampled arrays, and feature names.
    2. Call run_experiment() once per configured run index.
    3. Aggregate the returned metrics in the workflow module.

Outputs:
    - Per-run artifacts below run_XX_seed_YY in the configured output directory.
    - Scalar metrics dictionary returned to the workflow.

TODOs:
    - None identified.

Dependencies:
    - joblib.
    - numpy.
    - pandas.
    - scikit-learn.
    - tensorflow.
    - cnn_lstm_ddos_detection configuration, preprocessing, model, evaluation, persistence, system, and timing modules.

Assumptions & Notes:
    - Validation and held-out test data remain unaugmented throughout the run.
    - The best checkpoint is selected by validation accuracy exactly as in the original implementation.
================================================================================
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import PAPER_12_CLASSES
from .evaluation import metrics_from_predictions, save_confusion
from .model import MetalSafeDenseReLU, build_model, make_tf_dataset
from .persistence import persist_generated_dataset
from .preprocessing import apply_smote, preprocess_arrays, split_indices
from .system import set_seeds
from .timing import create_epoch_resource_logger, format_duration


@dataclass
class PreparedRunData:
    """Store transformed arrays, labels, indices, transformers, and preprocessing metadata for one run."""

    X_train_smote: np.ndarray
    y_train_smote: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_before_smote: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    imputer: SimpleImputer
    scaler: StandardScaler
    smote_report: Dict[str, object]
    preprocessing_manifest: Dict[str, object]


@dataclass
class ModelArrays:
    """Store sequence-shaped model inputs and one-hot targets for one experiment run."""

    X_train_model: np.ndarray
    X_val_model: np.ndarray
    X_test_model: np.ndarray
    y_train_onehot: np.ndarray
    y_val_onehot: np.ndarray
    y_test_onehot: np.ndarray


@dataclass
class TrainingArtifacts:
    """Store a trained model, training history, TensorFlow datasets, and training duration."""

    model: tf.keras.Model
    history: tf.keras.callbacks.History
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    training_seconds: float


@dataclass
class EvaluationArtifacts:
    """Store held-out evaluation outputs needed for metrics and persisted predictions."""

    test_loss: float
    keras_accuracy: float
    probabilities: np.ndarray
    predictions: np.ndarray
