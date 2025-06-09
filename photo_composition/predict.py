import argparse
from pathlib import Path

import torch
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

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_names = args.class_names.split(',')
    model = load_model(args.model, len(class_names), device)

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        best_class = probs.argmax().item()

    print(f"Predicted composition: {class_names[best_class]} (confidence {probs[best_class].item():.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict photo composition")
    parser.add_argument("--model", type=Path, required=True, help="Trained model file")
    parser.add_argument("--class-names", type=str, required=True, help="Comma-separated list of class names")
    parser.add_argument("--image", type=Path, required=True, help="Image file to evaluate")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    main(args)
