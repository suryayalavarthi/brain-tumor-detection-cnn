"""Grad-CAM visualization for brain tumor MRI classification."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import build_model

def _load_and_preprocess_image(
    image_path: str | Path, image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and prepare it for inference.

    Args:
        image_path: Path to the MRI image.
        image_size: Target size for resizing (width, height).

    Returns:
        A tuple containing:
            - Original RGB image resized to image_size.
            - Preprocessed image batch for the model.

    Raises:
        FileNotFoundError: If the image path does not exist.
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
    image_float = image.astype(np.float32) / 255.0
    batch = np.expand_dims(image_float, axis=0)
    return image, batch


def _get_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model.")


def _compute_gradcam(
    model: tf.keras.Model,
    image_batch: np.ndarray,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the top predicted class.

    Args:
        model: Trained Keras model.
        image_batch: Preprocessed image batch (1, H, W, C).

    Returns:
        Normalized heatmap as a 2D numpy array.
    """
    _ = model(image_batch, training=False)
    last_conv = _get_last_conv_layer(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.outputs[0]],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        tape.watch(conv_outputs)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Failed to compute Grad-CAM gradients.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    if np.max(heatmap) == 0:
        return heatmap
    return heatmap / np.max(heatmap)


def _overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay the heatmap on the original image.

    Args:
        image: Original RGB image (H, W, 3).
        heatmap: Normalized heatmap (H, W).
        alpha: Blend factor for the heatmap.

    Returns:
        RGB image with heatmap overlay.
    """
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def visualize_gradcam(
    image_path: str | Path,
    model_path: str | Path = "brain_tumor_model.keras",
    output_dir: str | Path = ".",
    show: bool = False,
) -> None:
    """Visualize Grad-CAM for a single MRI image.

    Args:
        image_path: Path to the MRI image.
        model_path: Path to the saved Keras model.
        output_dir: Directory to save output images.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_model = tf.keras.models.load_model(str(model_path), compile=False)
    model = build_model()
    model.set_weights(saved_model.get_weights())
    image, batch = _load_and_preprocess_image(image_path)
    heatmap = _compute_gradcam(model, batch)
    overlay = _overlay_heatmap(image, heatmap, alpha=0.4)

    plt.imsave(
        output_path / "sample_mri.png",
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        cmap="gray",
    )
    plt.imsave(
        output_path / "gradcam_heatmap.png",
        cv2.resize(heatmap, (image.shape[1], image.shape[0])),
        cmap="jet",
    )
    plt.imsave(output_path / "gradcam_overlay.png", overlay)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("The Explanation")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM explanation utility.")
    parser.add_argument("--image", required=True, help="Path to MRI image.")
    parser.add_argument(
        "--model", default="brain_tumor_model.keras", help="Path to model file."
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Output directory for Grad-CAM images.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Grad-CAM figure window.",
    )
    args = parser.parse_args()

    visualize_gradcam(args.image, args.model, args.out, args.show)
