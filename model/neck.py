import torch
from torch import nn


class TopDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.TopDown1 = TopDown(in_channels=1024, out_channels=512)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.TopDown2 = TopDown(in_channels=512 + 256, out_channels=256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.TopDown3 = TopDown(in_channels=256 + 128, out_channels=128)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        features_small, features_medium, features_large = x

        out1 = self.TopDown1(features_large)
        p1 = self.upsample(self.conv1(out1))
        p1 = torch.cat((p1, features_medium), dim=1)

        out2 = self.TopDown2(p1)
        p2 = self.upsample(self.conv2(out2))
        p2 = torch.cat((p2, features_small), dim=1)

        out3 = self.TopDown3(p2)

        return out1, out2, out3
