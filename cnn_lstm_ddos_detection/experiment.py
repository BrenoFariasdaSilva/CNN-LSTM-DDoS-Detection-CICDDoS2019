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


def build_preprocessing_manifest(feature_names: Sequence[str], feature_count: int, preprocessing_timings: Mapping[str, float], smote_report: Mapping[str, object]) -> Dict[str, object]:
    """
    Build the preprocessing manifest persisted for each experiment run.

    :param feature_names: Ordered readable feature names used by the model.
    :param feature_count: Number of transformed model features.
    :param preprocessing_timings: Measured preprocessing stage durations.
    :param smote_report: Training-only SMOTE audit metadata.
    :return: Preprocessing manifest preserving the original field structure.
    """

    return {
        "raw_dataset_modified": False,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "imputation": "median; fitted on training only",
        "standardization": "StandardScaler z-score; fitted on training only",
        "smote": dict(smote_report),
        "validation_augmented": False,
        "test_augmented": False,
        "feature_count": int(feature_count),
        "feature_names": list(feature_names),
        "timings": dict(preprocessing_timings),
        "paper_method_notes": {
            "class_mapping": "12-class mapping is inferred because the paper does not enumerate its 12 labels.",
            "sequence_construction": "Feature-axis CNN/LSTM sequence is a reconstruction because no temporal window length/stride is reported.",
            "architecture_widths": "Layer widths/kernel sizes are configurable reconstruction choices because the paper omits them.",
            "smote_parameters": "Paper mentions SMOTE but does not publish k or sampling strategy; classic k-neighbor SMOTE is used here.",
        },
    }  # Preserve the original preprocessing-manifest content and wording


def split_run_labels(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sampled labels and materialize integer label vectors for one run.

    :param y: Integer labels for the complete sampled real-data dataset.
    :param seed: Random seed used for the run's stratified split.
    :return: Train/validation/test indices followed by their aligned integer labels.
    """

    split_started = time.time()  # Start split-stage duration measurement
    train_idx, val_idx, test_idx = split_indices(y, seed)  # Create the run-specific 70/15/15 stratified partition
    y_train = y[train_idx].astype(np.int16, copy=False)  # Materialize training integer labels without unnecessary copying
    y_val = y[val_idx].astype(np.int16, copy=False)  # Materialize validation integer labels
    y_test = y[test_idx].astype(np.int16, copy=False)  # Materialize held-out test integer labels
    print(
        f"[SPLIT] train={len(y_train):,} (70%) val={len(y_val):,} (15%) "
        f"test={len(y_test):,} (15%) | completed in {format_duration(time.time()-split_started)}"
    )  # Report split sizes and duration using the original format
    return train_idx, val_idx, test_idx, y_train, y_val, y_test  # Return indices and aligned integer labels


def prepare_run_data(X: np.ndarray, y: np.ndarray, cfg: Config, seed: int, feature_names: Sequence[str], generated_dir: Path) -> PreparedRunData:
    """
    Split, preprocess, SMOTE-balance, optionally persist, and package one run's data.

    :param X: Complete sampled real-data feature matrix.
    :param y: Complete sampled integer-label vector.
    :param cfg: Validated experiment configuration.
    :param seed: Current experiment run seed.
    :param feature_names: Ordered readable feature names.
    :param generated_dir: Destination for optionally persisted transformed dataset copies.
    :return: PreparedRunData containing transformed arrays and preprocessing state.
    """

    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_run_labels(y, seed)  # Create this run's real-data partitions
    X_train, X_val, X_test, imputer, scaler, preprocessing_timings = preprocess_arrays(X, train_idx, val_idx, test_idx)  # Fit train-only imputation/scaling and transform all partitions
    print(
        f"[MEMORY] standardized arrays: train={X_train.nbytes / 2**30:.2f} GiB "
        f"val={X_val.nbytes / 2**30:.2f} GiB test={X_test.nbytes / 2**30:.2f} GiB"
    )  # Report standardized partition memory use
    X_train_smote, y_train_smote, smote_report = apply_smote(
        X_train=X_train,
        y_train=y_train,
        k_neighbors=cfg.smote_k_neighbors,
        seed=seed + 17_003,
        generation_chunk=cfg.smote_generation_chunk,
        neighbor_query_chunk=cfg.smote_neighbor_query_chunk,
    )  # Apply the original seed-derived SMOTE procedure to training data only
    print(
        f"[MEMORY] SMOTE train={X_train_smote.nbytes / 2**30:.2f} GiB "
        f"({len(y_train_smote):,} rows)"
    )  # Report post-SMOTE training memory use and row count
    preprocessing_manifest = build_preprocessing_manifest(feature_names, X_train.shape[1], preprocessing_timings, smote_report)  # Build auditable preprocessing metadata before releasing the pre-SMOTE training array
    if cfg.save_derived_data:  # Verify if transformed dataset persistence is enabled for this run
        persist_generated_dataset(
            generated_dir=generated_dir,
            X_train_smote=X_train_smote,
            y_train_smote=y_train_smote,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            manifest=preprocessing_manifest,
        )  # Persist the same transformed data copies and split indices as the original script
    if X_train_smote is not X_train:  # Verify if SMOTE created a new training feature matrix
        del X_train  # Release the pre-SMOTE standardized training matrix while preserving the resampled matrix
    gc.collect()  # Encourage prompt memory release before one-hot/model preparation
    return PreparedRunData(
        X_train_smote=X_train_smote,
        y_train_smote=y_train_smote,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_train_before_smote=y_train,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        imputer=imputer,
        scaler=scaler,
        smote_report=smote_report,
        preprocessing_manifest=preprocessing_manifest,
    )  # Return transformed data and fitted preprocessing state for model execution


def create_model_arrays(data: PreparedRunData) -> ModelArrays:
    """
    Add the feature-sequence channel axis and one-hot encode all three target partitions.

    :param data: Prepared transformed run data.
    :return: Sequence-shaped feature tensors and one-hot target matrices.
    """

    X_train_model = data.X_train_smote[..., None]  # Treat feature positions as the model sequence axis with one scalar channel
    X_val_model = data.X_val[..., None]  # Apply the same sequence representation to validation features
    X_test_model = data.X_test[..., None]  # Apply the same sequence representation to held-out test features
    class_count = len(PAPER_12_CLASSES)  # Use the fixed 12-class reconstruction size
    y_train_onehot = tf.keras.utils.to_categorical(data.y_train_smote, num_classes=class_count).astype(np.float32, copy=False)  # One-hot encode SMOTE-balanced training labels
    y_val_onehot = tf.keras.utils.to_categorical(data.y_val, num_classes=class_count).astype(np.float32, copy=False)  # One-hot encode unaugmented validation labels
    y_test_onehot = tf.keras.utils.to_categorical(data.y_test, num_classes=class_count).astype(np.float32, copy=False)  # One-hot encode unaugmented held-out test labels
    return ModelArrays(X_train_model, X_val_model, X_test_model, y_train_onehot, y_val_onehot, y_test_onehot)  # Return all model-ready arrays


def save_model_summary(model: tf.keras.Model, run_dir: Path) -> None:
    """
    Persist the Keras model summary for one experiment run.

    :param model: Compiled Keras model whose summary should be written.
    :param run_dir: Current run output directory.
    :return: None.
    """

    with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:  # Open the per-run summary output file
        model.summary(print_fn=lambda line: handle.write(line + "\n"))  # Write Keras summary lines exactly as generated by the model


def build_callbacks(cfg: Config, run_dir: Path) -> List[tf.keras.callbacks.Callback]:
    """
    Build checkpoint, CSV history, resource, and optional early-stopping callbacks.

    :param cfg: Validated experiment configuration.
    :param run_dir: Current run output directory.
    :return: Ordered Keras callback list used during model fitting.
    """

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
        create_epoch_resource_logger(cfg.epochs),
    ]  # Preserve checkpoint, CSV logger, and resource callback ordering
    if cfg.patience > 0:  # Verify if optional validation-accuracy early stopping is enabled
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=cfg.patience,
                restore_best_weights=True,
                verbose=1,
            )
        )  # Append the original optional early-stopping configuration
    return callbacks  # Return the ordered callback list


def train_model(cfg: Config, device: str, seed: int, run_dir: Path, arrays: ModelArrays) -> TrainingArtifacts:
    """
    Build datasets/model, fit with validation, and reload the best validation checkpoint.

    :param cfg: Validated experiment configuration.
    :param device: TensorFlow device selected for the experiment.
    :param seed: Current run seed used by training-data shuffling.
    :param run_dir: Current run output directory.
    :param arrays: Model-ready sequence inputs and one-hot targets.
    :return: Trained model, history, datasets, and measured training duration.
    """

    class_count = len(PAPER_12_CLASSES)  # Use the fixed output-class count for the reconstruction
    with tf.device(device):  # Build model variables on the selected TensorFlow device
        model = build_model(cfg, n_features=arrays.X_train_model.shape[1], n_classes=class_count)  # Construct and compile the configured CNN-LSTM model
    save_model_summary(model, run_dir)  # Persist the model architecture before training starts
    train_ds = make_tf_dataset(arrays.X_train_model, arrays.y_train_onehot, cfg.batch_size, True, seed)  # Build bounded-shuffle training batches
    val_ds = make_tf_dataset(arrays.X_val_model, arrays.y_val_onehot, cfg.batch_size, False, seed)  # Build deterministic validation batches
    test_ds = make_tf_dataset(arrays.X_test_model, arrays.y_test_onehot, cfg.batch_size, False, seed)  # Build deterministic held-out test batches
    callbacks = build_callbacks(cfg, run_dir)  # Build the original validation/checkpoint callback stack
    train_started = time.time()  # Start measured model-fitting duration
    with tf.device(device):  # Execute model fitting on the selected TensorFlow device
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=2,
        )  # Train with the unaugmented validation partition exactly as before
    training_seconds = time.time() - train_started  # Record measured model-training duration
    best_model_path = run_dir / "best_model.keras"  # Resolve the validation-best checkpoint path
    if best_model_path.exists():  # Verify if checkpointing produced a saved best model
        model = tf.keras.models.load_model(
            best_model_path,
            custom_objects={"MetalSafeDenseReLU": MetalSafeDenseReLU},
        )  # Reload the validation-best model before held-out evaluation
    return TrainingArtifacts(model, history, train_ds, val_ds, test_ds, training_seconds)  # Return fitted model state and datasets
