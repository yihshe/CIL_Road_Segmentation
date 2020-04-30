"""
UNet model structure.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Define UNet structure, with down and up phases.
    Can be configured with 2 or 3 layers, and different channel sizes.
    One must be careful of useful compatible channel sizes with provided images.
    All output values are passed into the sigmoid fonction for a binary classification.
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32, dropout=0.5):
        super(UNet, self).__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=dropout*0.5)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=dropout)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=dropout)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=dropout)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv6 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.dropout6 = nn.Dropout(p=dropout)
        self.decoder6 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv7 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.dropout7 = nn.Dropout(p=dropout)
        self.decoder7 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv8 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.dropout8 = nn.Dropout(p=dropout)
        self.decoder8 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv9 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.dropout9 = nn.Dropout(p=dropout)
        self.decoder9 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        p1 = self.pool1(enc1)
        p1 = self.dropout1(p1)

        enc2 = self.encoder2(p1)
        p2 = self.pool2(enc2)
        p2 = self.dropout2(p2)

        enc3 = self.encoder3(p2)
        p3 = self.pool3(enc3)
        p3 = self.dropout3(p3)

        enc4 = self.encoder4(p3)
        p4 = self.pool4(enc4)
        p4 = self.dropout4(p4)

        bottleneck = self.bottleneck(p4)

        u6 = self.upconv6(bottleneck)
        u6 = torch.cat((u6, enc4), dim=1)
        u6 = self.dropout6(u6)
        dec6 = self.decoder6(u6)

        u7 = self.upconv7(dec6)
        u7 = torch.cat((u7, enc3), dim=1)
        u7 = self.dropout7(u7)
        dec7 = self.decoder7(u7)

        u8 = self.upconv8(dec7)
        u8 = torch.cat((u8, enc2), dim=1)
        u8 = self.dropout8(u8)
        dec8 = self.decoder8(u8)

        u9 = self.upconv9(dec8)
        u9 = torch.cat((u9, enc1), dim=1)
        u9 = self.dropout9(u9)
        dec9 = self.decoder9(u9)

        return torch.sigmoid(self.conv(dec9))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
