"""Model architecture for brain tumor classification."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
) -> tf.keras.Model:
    """Create the CNN model architecture.

    Args:
        input_shape: Shape of the input images.
        num_classes: Number of output classes.

    Returns:
        Uncompiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_model(
    model: tf.keras.Model, learning_rate: float = 1e-4
) -> tf.keras.Model:
    """Compile a CNN model with Adam optimizer and categorical loss.

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Compiled Keras model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
