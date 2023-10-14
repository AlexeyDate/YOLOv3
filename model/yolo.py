import torch
from torch import nn
from model.backbone import Darknet53
from model.neck import FPN
from model.head import YoloHead


class YOLOv3(nn.Module):
    """
    Class implements the original architecture of the YOLOv3 model.
    This class contains Darknet53 model, it is CNN backbone of the YOLOv3 model.
    """

    def __init__(self, anchors, num_classes=1000, darknet_weights=None):
        super(YOLOv3, self).__init__()
        """
        param: anchors - anchor boxes
        param: num_classes - number of classes (default = 1000)
        param: darknet_weights - weight file of Darknet53 backbon model (default = None)
        """

        anchors = torch.tensor(anchors, dtype=torch.float32)

        self.backbone = Darknet53()
        self.neck = FPN()
        self.head = YoloHead(anchors=anchors, num_classes=num_classes)

        if darknet_weights is not None:
            self.backbone.load_weights(darknet_weights)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
