from torch import nn


class ResBlock(nn.Module):
    """
    Class implements residual block for Darknet53.
    It is contatins 2 consecutive сonvolutional layers.
    """
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        """
        param: in_channels - input channels for residual blocks 
        """

        assert in_channels % 2 == 0
        out_channels = in_channels // 2

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.1)
          )

    def forward(self, x):
        residual = x
        out = self.residual_block(x)
        out += residual
        return out
