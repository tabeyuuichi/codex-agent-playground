import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from model import CompositionNet
from dataset import create_test_loader


def load_model(model_path: Path, num_classes: int, device: torch.device) -> CompositionNet:
    model = CompositionNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# def evaluate(model: CompositionNet, loader: DataLoader, device: torch.device) -> float:
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return correct / total if total > 0 else 0.0


def evaluate(model, loader, device, class_names=None):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if class_names:
        report = classification_report(all_labels, all_preds, target_names=class_names)
    else:
        report = classification_report(all_labels, all_preds)
    print(report)
    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    return accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.class_names:
        class_names = args.class_names.split(',')
    else:
        classes_file = args.model.with_name(args.model.stem + "_classes.json")
        with classes_file.open() as f:
            class_names = json.load(f)

    loader, dataset_classes = create_test_loader(args.data_dir, args.image_size, args.batch_size, args.num_workers)

    model = load_model(args.model, len(class_names), device)

    acc = evaluate(model, loader, device)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate composition model on a test dataset")
    parser.add_argument("--model", type=Path, required=True, help="Trained model file")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to test dataset directory")
    parser.add_argument("--class-names", type=str, required=False, help="Comma-separated list of class names. If omitted, class names are loaded from <model>_classes.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    main(args)
