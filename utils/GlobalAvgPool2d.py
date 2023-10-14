from torch import nn


class GlobalAvgPool2d(nn.Module):
    """
    Class implements 2D global average pooling.
    It is necessary to train the classifier Darknet53 after convolutional layers.
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        height = x.data.size(2)
        width = x.data.size(3)
        layer = nn.AvgPool2d(kernel_size=height, stride=width)
        return layer(x)
