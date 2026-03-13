import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []

        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileFaceDetector(nn.Module):
    def __init__(self):
        super(MobileFaceDetector, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 128, 2, 6),
            InvertedResidual(128, 128, 1, 6),
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        self.shared_features = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), nn.ReLU6(inplace=True), nn.Dropout(0.2)
        )

        self.reg_head = nn.Linear(256, 8)
        self.conf_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.spatial_pool(x)
        x = self.flatten(x)

        features = self.shared_features(x)
        coords = self.reg_head(features)
        conf_logits = self.conf_head(features)
        return coords, conf_logits
