import os
from typing import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def create_dataloaders(data_dir: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, list]:
    """Create training and validation dataloaders.

    Args:
        data_dir: Dataset root directory. Expected structure is
            data_dir/train/<class_name>/image.png
            data_dir/val/<class_name>/image.png
        image_size: Resize size for the images.
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        train_loader, val_loader, class_names
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes
