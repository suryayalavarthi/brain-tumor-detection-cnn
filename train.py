"""Training script for brain tumor classification model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_data
from model import build_model, compile_model


def _save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training history as JSON.

    Args:
        history: Keras training history object.
        output_path: Path to save the JSON file.
    """
    history_data: Dict[str, Any] = history.history
    output_path.write_text(json.dumps(history_data, indent=2))


def _plot_training_curves(
    history: tf.keras.callbacks.History, output_path: Path
) -> None:
    """Plot and save training accuracy and loss curves.

    Args:
        history: Keras training history.
        output_path: Path to save the plot image.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Execute training workflow."""
    train_gen, val_gen, _ = load_data(dataset_dir="dataset")

    model = build_model()
    model = compile_model(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="brain_tumor_model.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    epochs = 10
    steps_per_epoch = min(100, math.ceil(train_gen.n / train_gen.batch_size))
    validation_steps = min(50, math.ceil(val_gen.n / val_gen.batch_size))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )

    _save_history(history, Path("training_history.json"))
    _plot_training_curves(history, Path("training_report.png"))


if __name__ == "__main__":
    main()
