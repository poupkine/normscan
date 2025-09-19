import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base_features, base_features*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base_features*2, base_features*4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(base_features*4, base_features*8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_features*8, base_features*4, 2, stride=2)
        self.dec3 = DoubleConv3D(base_features*8, base_features*4)
        self.up2 = nn.ConvTranspose3d(base_features*4, base_features*2, 2, stride=2)
        self.dec2 = DoubleConv3D(base_features*4, base_features*2)
        self.up1 = nn.ConvTranspose3d(base_features*2, base_features, 2, stride=2)
        self.dec1 = DoubleConv3D(base_features*2, base_features)

        self.final = nn.Conv3d(base_features, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))