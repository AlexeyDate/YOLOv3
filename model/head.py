import torch
from torch import nn


class Detect(nn.Module):
    def __init__(self, in_channels, anchors, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.num_anchors = anchors.size(0)
        self.num_classes = num_classes
        self.detect = nn.Conv2d(in_channels=in_channels * 2, out_channels=self.num_anchors * (5 + num_classes),
                                kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)
        predictions = self.detect(x)
        predictions = predictions.permute(0, 2, 3, 1)

        batch_size = predictions.size(0)
        s = predictions.size(1)
        predictions = predictions.view(batch_size, s, s, self.num_anchors, 5 + self.num_classes)

        predictions[..., 0] = torch.sigmoid(predictions[..., 0])
        predictions[..., 1:3] = torch.sigmoid(predictions[..., 1:3])

        # power method
        # predictions[..., 3:5] = torch.sigmoid(predictions[..., 3:5])

        return predictions


class YoloHead(nn.Module):
    def __init__(self, anchors, num_classes):
        super().__init__()

        self.detect_small = Detect(in_channels=128, anchors=anchors[0:3], num_classes=num_classes)
        self.detect_medium = Detect(in_channels=256, anchors=anchors[3:6], num_classes=num_classes)
        self.detect_large = Detect(in_channels=512, anchors=anchors[6:9], num_classes=num_classes)

    def forward(self, x):
        features_large, features_medium, features_small = x

        predict_small = self.detect_small(features_small)
        predict_medium = self.detect_medium(features_medium)
        predict_large = self.detect_large(features_large)

        return predict_small, predict_medium, predict_large
