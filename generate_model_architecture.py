"""Generate a professional model architecture diagram with plot_model."""

from __future__ import annotations

# If Graphviz isn't installed, run:
# brew install graphviz
# pip install pydot graphviz

from pathlib import Path

import tensorflow as tf

from model import build_model


def main() -> None:
    """Build model and save a layered architecture diagram."""
    model = build_model()

    output_dir = Path("visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_architecture.png"

    tf.keras.utils.plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,
        show_layer_names=True,
        expand_nested=False,
        dpi=200,
    )


if __name__ == "__main__":
    main()
