import argparse
from pathlib import Path
import json

import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

from model import CompositionNet


def load_model(model_path: Path, num_classes: int, device: torch.device) -> CompositionNet:
    model = CompositionNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rgb_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    sal_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    if args.class_names:
        class_names = args.class_names.split(',')
    else:
        classes_file = args.model.with_name(args.model.stem + "_classes.json")
        with classes_file.open() as f:
            class_names = json.load(f)
    model = load_model(args.model, len(class_names), device)

    image = Image.open(args.image).convert("RGB")
    with open(args.saliency, "rb") as f:
        sal_array = pickle.load(f)
    saliency = Image.fromarray(sal_array.astype(np.uint8)).convert("L")

    rgb_tensor = rgb_transform(image)
    saliency_tensor = sal_transform(saliency)
    tensor = torch.cat([rgb_tensor, saliency_tensor], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        best_class = probs.argmax().item()

    print(f"Predicted composition: {class_names[best_class]} (confidence {probs[best_class].item():.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict photo composition")
    parser.add_argument("--model", type=Path, required=True, help="Trained model file")
    parser.add_argument("--class-names", type=str, required=False, help="Comma-separated list of class names. If omitted, class names are loaded from <model>_classes.json")
    parser.add_argument("--image", type=Path, required=True, help="Image file to evaluate")
    parser.add_argument("--saliency", type=Path, required=True, help="Corresponding saliency map")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    main(args)
