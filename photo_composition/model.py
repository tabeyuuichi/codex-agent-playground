import torch
from torch import nn
from torchvision import models


class CompositionNet(nn.Module):
    """CNN model for photo composition classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        # Use a pretrained ResNet18 as the backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # conv1の入力チャネルを4に変更
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
