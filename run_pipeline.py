"""End-to-end training and evaluation pipeline (no notebooks)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from model import build_model, compile_model


def _plot_training_curves(history: tf.keras.callbacks.History) -> None:
    """Plot and save training accuracy and loss curves."""
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
    plt.savefig("training_report.png", dpi=200)
    plt.close()


def _evaluate(
    model: tf.keras.Model,
    val_gen: tf.keras.preprocessing.image.NumpyArrayIterator,
    class_mapping: dict[int, str],
) -> None:
    """Run evaluation and save confusion matrix."""
    val_predictions = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_gen.classes

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[class_mapping[i] for i in sorted(class_mapping)],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_mapping[i] for i in sorted(class_mapping)],
        yticklabels=[class_mapping[i] for i in sorted(class_mapping)],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()


def _predict_single(
    model: tf.keras.Model,
    image_path: str,
    class_mapping: dict[int, str],
) -> None:
    """Predict a single image and print the result."""
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    prediction = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index]) * 100.0
    label = class_mapping.get(class_index, "unknown")
    print(f"Prediction: {label} ({confidence:.2f}%)")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Brain tumor CNN pipeline.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-val-steps", type=int, default=50)
    parser.add_argument("--predict", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run training, evaluation, and optional prediction."""
    args = parse_args()
    train_gen, val_gen, class_mapping = load_data(dataset_dir=args.dataset_dir)

    model = compile_model(build_model())

    steps_per_epoch = min(
        args.max_steps, math.ceil(train_gen.n / train_gen.batch_size)
    )
    validation_steps = min(
        args.max_val_steps, math.ceil(val_gen.n / val_gen.batch_size)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="brain_tumor_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )

    Path("training_history.json").write_text(
        json.dumps(history.history, indent=2)
    )
    _plot_training_curves(history)
    _evaluate(model, val_gen, class_mapping)

    if args.predict:
        _predict_single(model, args.predict, class_mapping)


if __name__ == "__main__":
    main()
