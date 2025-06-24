import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True) if activation == "RE" else HardSwish()
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True) if activation == "RE" else HardSwish()
        ]
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )

        self.blocks = nn.Sequential(
            MobileBottleneck(16, 16, 3, 1, 1, False, "RE"),
            MobileBottleneck(16, 24, 3, 2, 4, False, "RE"),
            MobileBottleneck(24, 24, 3, 1, 3, False, "RE"),
            MobileBottleneck(24, 40, 5, 2, 3, True, "RE"),
            MobileBottleneck(40, 40, 5, 1, 3, True, "RE"),
            MobileBottleneck(40, 40, 5, 1, 3, True, "RE"),
            MobileBottleneck(40, 80, 3, 2, 6, False, "HS"),
            MobileBottleneck(80, 80, 3, 1, 2.5, False, "HS"),
            MobileBottleneck(80, 80, 3, 1, 2.3, False, "HS"),
            MobileBottleneck(80, 80, 3, 1, 2.3, False, "HS"),
            MobileBottleneck(80, 112, 3, 1, 6, True, "HS"),
            MobileBottleneck(112, 112, 3, 1, 6, True, "HS"),
            MobileBottleneck(112, 160, 5, 2, 6, True, "HS"),
            MobileBottleneck(160, 160, 5, 1, 6, True, "HS"),
            MobileBottleneck(160, 160, 5, 1, 6, True, "HS"),
        )

        self.final = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            HardSwish()
        )

        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.classifier(x)
        return x