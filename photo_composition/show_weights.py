"""Utility to load a trained composition model and display its weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import CompositionNet


def load_model(weights_path: Path, classes_path: Path) -> CompositionNet:
    """Load a CompositionNet instance with the given weights."""
    with classes_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = CompositionNet(num_classes=len(class_names))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Show weights of a trained model")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("composition_model.pth"),
        help="Path to the trained model weights (.pth file).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("composition_model_classes.json"),
        help="Path to the JSON file containing class names.",
    )
    args = parser.parse_args()

    model = load_model(args.weights, args.classes)
    torch.set_printoptions(precision=6, sci_mode=False)

    print(f"Loaded weights from: {args.weights}")
    print(f"Loaded class definitions from: {args.classes}")
    print("\nModel parameters:")
    for name, param in model.state_dict().items():
        print(f"\n{name} (shape={tuple(param.shape)}):\n{param}")


if __name__ == "__main__":
    main()
