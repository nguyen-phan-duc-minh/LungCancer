import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        inverted_residual_setting = [
            # t, c, n, s
            [3, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        self.drop = nn.Dropout(0.6)
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.classifier(x)
        return x