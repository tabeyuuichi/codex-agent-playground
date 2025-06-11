import os
import pickle
import numpy as np
import torch
from typing import Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

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

    # 前処理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # クラス一覧取得のため一時的に ImageFolder を使用
    temp = datasets.ImageFolder(train_dir)
    class_to_idx = temp.class_to_idx
    class_names = temp.classes

    # ImageFolder は、フォルダ名をラベルとして認識する自動ラベリングクラス
    train_dataset = FourChannelImageFolder(train_dir, transform, class_to_idx)
    val_dataset = FourChannelImageFolder(val_dir, transform, class_to_idx)

    # DataLoader に変換（バッチ化）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names

def create_test_loader(data_dir: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, list]:
    """Create a dataloader for a test dataset.

    Args:
        data_dir: Path to the test dataset directory containing class subfolders.
        image_size: Resize size for the images.
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker processes for data loading.

    Returns:
        test_loader, class_names
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    temp = datasets.ImageFolder(data_dir)
    class_to_idx = temp.class_to_idx
    class_names = temp.classes

    dataset = FourChannelImageFolder(data_dir, transform, class_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, class_names


class FourChannelImageFolder(Dataset):
    def __init__(self, image_dir, class_to_idx, transform=None):
        self.image_dir = image_dir
        self.saliency_dir = 'saliency'
        self.class_to_idx = class_to_idx  # {class_name: idx}
        self.transform = transform
        self.samples = []
        for class_name in os.listdir(image_dir):
            class_path = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                file_path = os.path.join(class_path, fname)
                if os.path.isdir(file_path):  # e.g. "saliency" directory
                    continue
                img_path = file_path
                base, _ = os.path.splitext(fname)
                sal_fname = base + '.pickle'
                sal_path = os.path.join(class_path, self.saliency_dir, sal_fname)  # 拡張子を .pickle に変更
                self.samples.append((img_path, sal_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sal_path, label = self.samples[idx]
        rgb = Image.open(img_path).convert("RGB")
        with open(sal_path, "rb") as f:
            sal_array = pickle.load(f)
        sal = Image.fromarray(sal_array.astype(np.uint8)).convert("L")  # 1チャネル

        if self.transform:
            rgb = self.transform(rgb)        # (3, H, W)
            sal = self.transform(sal)        # (1, H, W)

        img_4ch = torch.cat([rgb, sal], dim=0)  # (4, H, W)
        return img_4ch, label
