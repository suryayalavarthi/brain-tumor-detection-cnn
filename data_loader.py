"""Data loading utilities for brain tumor MRI dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

LOGGER = logging.getLogger(__name__)


def load_data(
    dataset_dir: str | Path = "dataset",
    image_size: Tuple[int, int] = (224, 224),
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
) -> Tuple[
    tf.keras.preprocessing.image.NumpyArrayIterator,
    tf.keras.preprocessing.image.NumpyArrayIterator,
    Dict[int, str],
]:
    """Load MRI images, split dataset, and create generators.

    Args:
        dataset_dir: Root directory containing class subfolders.
        image_size: Target size for resizing images (width, height).
        test_size: Fraction of data reserved for validation.
        random_state: Random seed for reproducibility.
        batch_size: Batch size for generators.

    Returns:
        Training generator, validation generator, and class index mapping.

    Raises:
        FileNotFoundError: If dataset directory does not exist or is empty.
        RuntimeError: If no valid images are found.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        fallback_path = Path("data")
        if fallback_path.exists() and fallback_path.is_dir():
            LOGGER.warning(
                "Dataset directory '%s' not found. Falling back to '%s'.",
                dataset_path,
                fallback_path,
            )
            dataset_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path.resolve()}"
            )

    archive_path = dataset_path / "archive"
    if archive_path.is_dir() and (archive_path / "Training").is_dir():
        LOGGER.warning(
            "Found archive layout. Using dataset root: %s", archive_path
        )
        dataset_path = archive_path

    training_path = dataset_path / "Training"
    testing_path = dataset_path / "Testing"

    if training_path.is_dir():
        class_names = sorted(
            [p.name for p in training_path.iterdir() if p.is_dir()]
        )
    else:
        class_names = sorted(
            [
                p.name
                for p in dataset_path.iterdir()
                if p.is_dir()
            ]
        )

    if not class_names:
        raise FileNotFoundError(
            f"No class subfolders found in: {dataset_path.resolve()}"
        )

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}

    filepaths: List[str] = []
    labels: List[str] = []
    valid_suffixes = {".jpg", ".jpeg", ".png"}

    if training_path.is_dir():
        sources = [training_path]
        if testing_path.is_dir():
            sources.append(testing_path)
        for class_name in class_names:
            for source in sources:
                class_path = source / class_name
                if not class_path.is_dir():
                    continue
                for image_path in class_path.iterdir():
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower() not in valid_suffixes:
                        continue
                    try:
                        image = cv2.imread(str(image_path))
                        if image is None:
                            raise ValueError(
                                f"Unable to read image: {image_path}"
                            )
                        filepaths.append(str(image_path))
                        labels.append(class_name)
                    except ValueError as exc:
                        LOGGER.warning("Skipping image %s: %s", image_path, exc)
    else:
        for class_name in class_names:
            class_path = dataset_path / class_name
            for image_path in class_path.iterdir():
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in valid_suffixes:
                    continue
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"Unable to read image: {image_path}")
                    filepaths.append(str(image_path))
                    labels.append(class_name)
                except ValueError as exc:
                    LOGGER.warning("Skipping image %s: %s", image_path, exc)

    if not filepaths:
        raise RuntimeError("No valid images found in dataset.")

    x_train, x_val, y_train, y_val = train_test_split(
        filepaths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_df = pd.DataFrame({"filepath": x_train, "label": y_train})
    val_df = pd.DataFrame({"filepath": x_val, "label": y_val})

    train_generator = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=image_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        classes=class_names,
    )
    validation_generator = val_gen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=image_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        classes=class_names,
    )

    return train_generator, validation_generator, index_to_class
