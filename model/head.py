import torch
from torch import contiguous_format, nn


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
        
        batch_size = x.size(0)
        s = x.size(2) # Getting height or width from outputs

        # out(bs, 3 * (num_classes + 5), height, width) to out(bs, height, width, 3, 5 + num_classes)
        predictions = self.detect(x).view(batch_size, self.num_anchors, self.num_classes + 5, s, s)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()

        return predictions


class YoloHead(nn.Module):
    def __init__(self, anchors, num_classes):
        super().__init__()

        self.detect_small = Detect(in_channels=128, anchors=anchors[2], num_classes=num_classes)
        self.detect_medium = Detect(in_channels=256, anchors=anchors[1], num_classes=num_classes)
        self.detect_large = Detect(in_channels=512, anchors=anchors[0], num_classes=num_classes)

    def forward(self, x):
        features_large, features_medium, features_small = x

        predict_small = self.detect_small(features_small)
        predict_medium = self.detect_medium(features_medium)
        predict_large = self.detect_large(features_large)

        return predict_large, predict_medium, predict_small 
