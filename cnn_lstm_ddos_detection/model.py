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

    def build(self: "MetalSafeDenseReLU", input_shape: Any) -> None:
        """
        Create trainable kernel and bias weights after the input feature size is known.

        :param self: Current MetalSafeDenseReLU layer instance.
        :param input_shape: TensorFlow/Keras input shape supplied during layer building.
        :return: None.
        """

        input_dim = int(input_shape[-1])  # Read the flattened input width used by the dense transform
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )  # Create the trainable dense kernel with the original initializer
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )  # Create the trainable dense bias with the original initializer
        super().build(input_shape)  # Mark the Keras layer as built through the base implementation

    def call(self: "MetalSafeDenseReLU", inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the Metal-safe dense affine transform followed by ReLU.

        :param self: Current MetalSafeDenseReLU layer instance.
        :param inputs: Input tensor for the dense transformation.
        :return: ReLU-activated dense output tensor.
        """

        transformed = tf.linalg.matmul(inputs, self.kernel)  # Apply the dense matrix multiplication explicitly
        transformed = tf.add(transformed, self.bias)  # Add bias with AddV2 instead of TensorFlow BiasAdd
        return tf.nn.relu(transformed)  # Apply the same ReLU activation used by the original custom layer

    def get_config(self: "MetalSafeDenseReLU") -> Dict[str, Any]:
        """
        Return serializable Keras configuration for this custom layer.

        :param self: Current MetalSafeDenseReLU layer instance.
        :return: Base Keras layer configuration extended with the dense unit count.
        """

        config = super().get_config()  # Retrieve standard Keras layer serialization fields
        config.update({"units": self.units})  # Persist the custom dense width required for deserialization
        return config  # Return the complete serializable layer configuration

    @classmethod
    def from_config(cls: Type["MetalSafeDenseReLU"], config: Dict[str, Any]) -> "MetalSafeDenseReLU":
        """
        Reconstruct the custom layer from its serialized Keras configuration.

        :param cls: MetalSafeDenseReLU class used to reconstruct the layer.
        :param config: Serialized layer configuration containing the custom unit count.
        :return: Reconstructed MetalSafeDenseReLU layer.
        """

        restored = dict(config)  # Copy the serialized configuration so the caller's mapping is not mutated
        units = int(restored.pop("units"))  # Remove and preserve the custom dense width before base-layer construction
        layer = cls(**restored)  # Construct the layer through the inherited Keras Layer initializer
        layer.units = units  # Restore the custom dense width before Keras builds the layer
        return layer  # Return the deserializable custom layer instance


def create_metal_safe_dense_relu(units: int, name: Optional[str] = None) -> MetalSafeDenseReLU:
    """
    Create an initialized MetalSafeDenseReLU layer without defining a private constructor.

    :param units: Number of output units produced by the dense transformation.
    :param name: Optional Keras layer name.
    :return: Initialized custom dense/ReLU layer.
    """

    layer = MetalSafeDenseReLU(name=name)  # Construct the custom layer through the inherited public Keras initializer
    layer.units = int(units)  # Store the requested output width before the first build call
    return layer  # Return the initialized Metal-safe custom layer


def build_optimizer(cfg: Config) -> tf.keras.optimizers.Optimizer:
    """
    Build the configured optimizer using the original Adam-or-SGD selection logic.

    :param cfg: Validated experiment configuration.
    :return: Configured TensorFlow/Keras optimizer instance.
    """

    if cfg.optimizer == "adam":  # Verify if Adam is the configured optimizer
        return tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)  # Create Adam with the configured learning rate
    return tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate)  # Preserve SGD as the only alternative optimizer path


def build_model(cfg: Config, n_features: int, n_classes: int) -> tf.keras.Model:
    """
    Build and compile the reconstructed CNN with parallel Dense and LSTM branches.

    :param cfg: Validated experiment and architecture configuration.
    :param n_features: Number of standardized feature positions per input row.
    :param n_classes: Number of output classes for softmax classification.
    :return: Compiled TensorFlow/Keras model.
    """

    inputs = tf.keras.Input(shape=(n_features, 1), dtype=tf.float32, name="traffic_features")  # Define one scalar channel over the feature-position sequence
    convolution = tf.keras.layers.Conv1D(
        cfg.conv_filters_1,
        cfg.kernel_size,
        padding="same",
        activation="relu",
        name="cnn_conv_1",
    )(inputs)  # Apply the first CNN feature-extraction layer
    convolution = tf.keras.layers.MaxPooling1D(cfg.pool_size, name="cnn_pool_1")(convolution)  # Apply the first configured max-pooling operation
    if cfg.conv_filters_2 > 0:  # Verify if the optional second convolutional layer is enabled
        convolution = tf.keras.layers.Conv1D(
            cfg.conv_filters_2,
            cfg.kernel_size,
            padding="same",
            activation="relu",
            name="cnn_conv_2",
        )(convolution)  # Apply the optional second CNN feature-extraction layer
        convolution = tf.keras.layers.MaxPooling1D(cfg.pool_size, name="cnn_pool_2")(convolution)  # Apply the second configured max-pooling operation
    dense_branch = tf.keras.layers.Flatten(name="dense_flatten")(convolution)  # Flatten CNN feature maps for the parallel dense branch
    dense_branch = create_metal_safe_dense_relu(cfg.dense_units, name="dense_branch")(dense_branch)  # Apply the Metal-safe dense/ReLU transformation
    if cfg.dropout > 0:  # Verify if branch dropout is enabled
        dense_branch = tf.keras.layers.Dropout(cfg.dropout, name="dense_dropout")(dense_branch)  # Apply dropout to the dense branch
    lstm_branch = tf.keras.layers.LSTM(cfg.lstm_units, name="lstm_branch")(convolution)  # Process CNN feature maps through the parallel LSTM branch
    if cfg.dropout > 0:  # Verify if branch dropout is enabled
        lstm_branch = tf.keras.layers.Dropout(cfg.dropout, name="lstm_dropout")(lstm_branch)  # Apply dropout to the LSTM branch
    merged = tf.keras.layers.Concatenate(name="parallel_concat")([dense_branch, lstm_branch])  # Concatenate outputs from the parallel Dense and LSTM branches
    if cfg.post_dense_units > 0:  # Verify if the optional post-concatenation dense layer is enabled
        merged = create_metal_safe_dense_relu(cfg.post_dense_units, name="post_merge_dense")(merged)  # Apply the configured Metal-safe post-merge dense layer
        if cfg.dropout > 0:  # Verify if post-merge dropout is enabled
            merged = tf.keras.layers.Dropout(cfg.dropout, name="post_merge_dropout")(merged)  # Apply dropout after the post-merge dense layer
    logits = tf.keras.layers.Dense(n_classes, activation=None, name="class_logits")(merged)  # Produce one unnormalized logit per target class
    outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="class_probabilities")(logits)  # Convert logits to float32 class probabilities
    model = tf.keras.Model(inputs, outputs, name="CNN_LSTM_DDoS_Detection_CICDDoS2019")  # Assemble the reconstructed model graph with the original model name
    model.compile(
        optimizer=build_optimizer(cfg),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )  # Compile with categorical cross-entropy and accuracy exactly as before
    return model  # Return the compiled model ready for fitting


def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    """
    Build the bounded-memory TensorFlow dataset used for training or evaluation.

    :param X: Model input feature tensor.
    :param y: One-hot target matrix aligned with X.
    :param batch_size: Number of samples per TensorFlow batch.
    :param training: Whether bounded shuffling should be enabled.
    :param seed: Random seed used by TensorFlow dataset shuffling.
    :return: Batched and single-batch-prefetched TensorFlow dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices((X, y))  # Create a TensorFlow dataset from aligned in-memory feature and target arrays
    if training:  # Verify if this dataset will be used for model fitting
        dataset = dataset.shuffle(
            buffer_size=min(len(y), 100_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )  # Shuffle training rows with the original bounded buffer and seed behavior
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Batch all rows without dropping the final partial batch
    return dataset.prefetch(1)  # Prefetch exactly one batch to limit unified-memory pressure
