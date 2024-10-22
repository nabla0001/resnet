"""Implements original ResNet CIFAR-10 architecture from He et al. 2015 https://arxiv.org/pdf/1512.03385"""

import torch.nn as nn
from torch import Tensor
from typing import Optional


"""Modules for the skip connection of residual blocks.

they handle the mismatch between x (forwarded via skip connection) and the residual output, e.g. when 
the previous layer outputs a feature map x of size (C=16, H=32, W=32) and the current layer produces a 
feature map f of size (C=32, H=16, W=16), x needs to be brought to (C=16, H=32, W=32) to be able element-wise add it to f.
"""
class ZeroPadding(nn.Module):
    """Option A in He et al. 2015"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_downsample = nn.MaxPool2d(kernel_size=1, stride=stride)
        self.pad = out_channels - in_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.spatial_downsample(x)  # ignores every 2nd pixel
        out = nn.functional.pad(out, (0, 0, 0, 0, 0, self.pad))  # pad along channel dimension
        return out


class Conv1x1Projection(nn.Module):
    """Option B in He et al. 2015"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        return out


class ZeroPaddingMaxPool(nn.Module):
    """Explored alternative to option A (not in He et al. 2015).
    spatial down-sampling is achieved by conventional max pooling with a
    2x2 kernel and stride of 2 instead of skipping pixels.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_downsample = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.pad = out_channels - in_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.spatial_downsample(x)
        out = nn.functional.pad(out, (0, 0, 0, 0, 0, self.pad))  # pad along channel dimension
        return out


# ResNet block
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 skip_connection: Optional[type[ZeroPadding | ZeroPaddingMaxPool | Conv1x1Projection]] = None):
        super().__init__()

        if skip_connection is None:
            self.skip_connection = None
        else:
            self.skip_connection = skip_connection(in_channels, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        out = out + identity

        out = self.relu2(out)
        return out


# PlainNet block
class PlainBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 skip_connection=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block: type[ResidualBlock | PlainBlock],
                 skip_connection: Optional[type[ZeroPadding | ZeroPaddingMaxPool | Conv1x1Projection]],
                 num_blocks: tuple[int, int, int],
                 num_classes: int = 10):
        super().__init__()
        self.in_channels = 16
        self.skip_connection = skip_connection

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)  # output: 16x32x32
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)  # output: 32x16x16
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)  # output: 64x8x8
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # initialise weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self,
                    block: nn.Module,
                    out_channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        blocks = []
        if (stride != 1) or (self.in_channels != out_channels):
            blocks.append(block(self.in_channels, out_channels, stride, self.skip_connection))
        else:
            blocks.append(block(self.in_channels, out_channels, stride, None))

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(block(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out