import torch
from torch import nn
from torchvision import models


class CompositionNet(nn.Module):
    """CNN model for photo composition classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        # Use a pretrained ResNet18 as the backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
