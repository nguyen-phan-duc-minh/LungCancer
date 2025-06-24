import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(squeezed_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super().__init__()
        self.stride = stride
        self.use_residual = (in_channels == out_channels and stride == 1)
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            SqueezeExcite(hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1),

            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, kernel_size=3, stride=1),

            MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=5, stride=1),

            MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1),

            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1),

            MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1),

            MBConvBlock(192, 320, expand_ratio=6, kernel_size=3, stride=1),
            MBConvBlock(320, 320, expand_ratio=6, kernel_size=3, stride=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x