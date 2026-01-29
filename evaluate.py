"""Evaluation utilities for brain tumor classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data


def _preprocess_image(
    image_path: str | Path, image_size: Tuple[int, int]
) -> np.ndarray:
    """Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        image_size: Target size for resizing (width, height).

    Returns:
        Preprocessed image array.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image cannot be read.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model,
    index_to_class: Dict[int, str],
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    """Predict the class for a single MRI image.

    Args:
        image_path: Path to the image.
        model: Trained Keras model.
        index_to_class: Mapping from class index to label.
        image_size: Target image size (width, height).
    """
    image = _preprocess_image(image_path, image_size)
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index]) * 100.0
    label = index_to_class.get(class_index, "unknown")
    print(f"Prediction: {label} ({confidence:.2f}%)")


def main() -> None:
    """Run evaluation on the validation split and save metrics."""
    _, val_gen, index_to_class = load_data(dataset_dir="dataset")
    model = tf.keras.models.load_model("brain_tumor_model.h5")

    val_predictions = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = val_gen.classes

    report = classification_report(
        y_true,
        y_pred,
        target_names=[index_to_class[i] for i in sorted(index_to_class)],
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[index_to_class[i] for i in sorted(index_to_class)],
        yticklabels=[index_to_class[i] for i in sorted(index_to_class)],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
