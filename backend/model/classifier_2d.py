# src/model/classifier_2d.py
import torch
import torch.nn as nn


class SimpleClassifier2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_features=32):
        super(SimpleClassifier2D, self).__init__()

        # Базовые блоки: Conv -> BatchNorm -> ReLU
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(in_channels, base_features),
            conv_block(base_features, base_features * 2),
            conv_block(base_features * 2, base_features * 4),
            conv_block(base_features * 4, base_features * 8),  # (B, 256, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 256, 1, 1)
            nn.Flatten(),                  # (B, 256)
            nn.Linear(base_features * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Выход в [0, 1] — вероятность патологии
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # (B,) - скалярная вероятность
