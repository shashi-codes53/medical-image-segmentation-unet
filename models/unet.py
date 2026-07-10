"""
U-Net Architecture for Medical Image Segmentation
Paper: https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv -> BN -> ReLU) x 2
    This is the basic building block used throughout U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder (Downsampling) Block:
    DoubleConv -> MaxPool
    Returns both the feature map (for skip connection) and the pooled output.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)       # save for skip connection
        pooled = self.pool(skip)  # downsample
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    Decoder (Upsampling) Block:
    Upsample -> Concat skip connection -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # in_channels because of concat

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch (if input not perfectly divisible)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([skip, x], dim=1)  # concat along channel axis
        return self.conv(x)


class UNet(nn.Module):
    """
    Full U-Net Architecture

    Args:
        in_channels  : number of input channels (1 for grayscale X-ray, 3 for RGB)
        out_channels : number of output classes (1 for binary segmentation)
        features     : list of feature sizes for each encoder level

    Architecture:
        Input
          └─ Encoder L1 (64)  ──────────────────────────────┐ skip1
              └─ Encoder L2 (128) ─────────────────────┐ skip2│
                  └─ Encoder L3 (256) ────────────┐ skip3│    │
                      └─ Encoder L4 (512) ────┐ skip4│   │    │
                          └─ Bottleneck (1024) │     │   │    │
                              └─ Decoder L4 ──┘     │   │    │
                                  └─ Decoder L3 ────┘   │    │
                                      └─ Decoder L2 ────┘    │
                                          └─ Decoder L1 ─────┘
                                              └─ Output Conv (1x1)
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat))
            prev_channels = feat

        # Bottleneck (deepest layer, no pooling)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path (reversed features)
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feat in rev_features:
            self.decoders.append(DecoderBlock(prev_channels, feat))
            prev_channels = feat

        # Final 1x1 conv to map to output classes
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder: collect skip connections
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: use skip connections in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)


def get_model(in_channels=1, out_channels=1):
    """Factory function to get U-Net model."""
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)  # batch=2, channels=1, 256x256
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")  # should be (2, 1, 256, 256)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
