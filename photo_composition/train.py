import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataset import create_dataloaders
from model import CompositionNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = create_dataloaders(args.data_dir, args.image_size, args.batch_size)

    model = CompositionNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {train_loss:.4f} val loss: {val_loss:.4f} val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train composition estimation model")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=Path, default="composition_model.pth", help="Output model file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    main(args)
