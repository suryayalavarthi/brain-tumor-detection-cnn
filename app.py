"""Streamlit app for brain tumor prediction with Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from model import build_model

MODEL_PATH = Path("brain_tumor_model.keras")
MODEL_URL = (
    "https://github.com/suryayalavarthi/brain-tumor-detection-cnn/"
    "releases/download/v1.0/brain_tumor_model.keras"
)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE: Tuple[int, int] = (224, 224)


def _load_image(uploaded_file: bytes) -> np.ndarray:
    """Decode uploaded file into RGB image."""
    file_bytes = np.asarray(bytearray(uploaded_file), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode image. Please upload a valid file.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Normalize and batch image for inference."""
    image = image_rgb.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def _get_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model.")


def _compute_gradcam(model: tf.keras.Model, image_batch: np.ndarray) -> np.ndarray:
    """Compute Grad-CAM heatmap for the top predicted class."""
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
    heatmap = tf.nn.relu(heatmap).numpy()
    return heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap


def _overlay_heatmap(
    image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)


def _encode_png(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image.")
    return buffer.tobytes()


def _label_badge(label: str) -> str:
    """Return a colored HTML badge for the predicted label."""
    color_map = {
        "notumor": "#1f9d55",
        "glioma": "#d97706",
        "meningioma": "#b91c1c",
        "pituitary": "#7c3aed",
    }
    color = color_map.get(label, "#0f172a")
    return (
        f"<span style='background:{color};color:white;"
        "padding:4px 10px;border-radius:999px;"
        "font-size:0.85rem;font-weight:600;'>"
        f"{label}</span>"
    )


@st.cache_resource(show_spinner=False)
def _load_model() -> tf.keras.Model:
    """Load trained weights into the architecture for Grad-CAM stability."""
    if not MODEL_PATH.exists():
        _download_model()
    saved_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    model = build_model()
    model.set_weights(saved_model.get_weights())
    return model


def _download_model() -> None:
    """Download the model file from GitHub Releases."""
    import urllib.request

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model weights..."):
        with urllib.request.urlopen(MODEL_URL) as response:
            total = int(response.headers.get("Content-Length", "0"))
            progress = st.progress(0)
            downloaded = 0
            chunk_size = 1024 * 1024
            with open(MODEL_PATH, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if total > 0:
                        downloaded += len(chunk)
                        progress.progress(min(downloaded / total, 1.0))
            progress.progress(1.0)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Brain Tumor Grad-CAM", layout="wide")
    st.title("Brain Tumor Classification with Grad‑CAM")
    st.caption(
        "Upload an MRI scan to get the model prediction and explainability map."
    )

    with st.sidebar:
        st.subheader("Model Info")
        st.write("Custom CNN, 4-class classification")
        st.write("Input: 224×224 RGB")
        st.write(f"Model file: `{MODEL_PATH.name}`")

    uploaded = st.file_uploader(
        "Upload MRI image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    try:
        image_rgb = _load_image(uploaded.read())
        image_batch = _preprocess(image_rgb)
        model = _load_model()

        preds = model.predict(image_batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100.0

        heatmap = _compute_gradcam(model, image_batch)
        overlay = _overlay_heatmap(image_rgb, heatmap, alpha=0.4)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Prediction")
            st.markdown(_label_badge(pred_label), unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")
            prob_dict = {name: float(score) for name, score in zip(CLASS_NAMES, preds)}
            st.bar_chart(prob_dict, height=200)
            st.image(image_rgb, caption="Uploaded MRI", use_column_width=True)
            st.download_button(
                "Download Original Image",
                data=_encode_png(image_rgb),
                file_name="mri_original.png",
                mime="image/png",
            )
        with right:
            st.subheader("Grad‑CAM Heatmap")
            tabs = st.tabs(["Overlay", "Heatmap"])
            with tabs[0]:
                st.image(
                    overlay,
                    caption="Model Focus (Overlay)",
                    use_column_width=True,
                )
            with tabs[1]:
                heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
                st.image(
                    heatmap_resized,
                    caption="Grad‑CAM Heatmap",
                    use_column_width=True,
                )
            st.download_button(
                "Download Grad‑CAM Overlay",
                data=_encode_png(overlay),
                file_name="gradcam_overlay.png",
                mime="image/png",
            )

        with st.expander("What does Grad‑CAM show?"):
            st.write(
                "Grad‑CAM highlights the regions that most influenced the "
                "model's prediction, helping clinicians verify that the "
                "model is focusing on relevant tumor features."
            )
    except Exception as exc:
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
