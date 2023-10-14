import numpy as np
from torch import nn
from utils.loadWeights import load_conv, load_conv_batch_norm
from utils.GlobalAvgPool2d import GlobalAvgPool2d
from utils.ResBlock import ResBlock


class Darknet53(nn.Module):
    """
    Base class of YOLOv3 architecture.
    It contains 53 convolutional layers and is also used in YOLOv3
    with pretrained weights on classification.

    Note: Use load_weights() in this class for better training
    """
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_block1 = nn.Sequential()
        for block in range(1):
            self.residual_block1.add_module(f"res{block}", ResBlock(in_channels=64))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.residual_block2 = nn.Sequential()
        for block in range(2):
            self.residual_block2.add_module(f"res{block}", ResBlock(in_channels=128))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_block3 = nn.Sequential()
        for block in range(8):
            self.residual_block3.add_module(f"res{block}", ResBlock(in_channels=256))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_block4 = nn.Sequential()
        for block in range(8):
            self.residual_block4.add_module(f"res{block}", ResBlock(in_channels=512))

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_block5 = nn.Sequential()
        for block in range(4):
            self.residual_block5.add_module(f"res{block}", ResBlock(in_channels=1024))

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=True),
            GlobalAvgPool2d(),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.residual_block1(x)
        x = self.conv_block2(x)
        x = self.residual_block2(x)
        x = self.conv_block3(x)
        out1 = self.residual_block3(x)

        x = self.conv_block4(out1)
        out2 = self.residual_block4(x)

        x = self.conv_block5(out2)
        out3 = self.residual_block5(x)

        return out1, out2, out3

        # Note: Use these lines if you want to train Darknet53 from scratch
        # x = self.classifier(out3)
        # return x

    def load_weights(self, weightfile):
        """
        Loading weights to Darknet53.
        You can download these weights on the follow links:
        https://pjreddie.com/media/files/darknet53_448.weights

        param weightfile: binary file of weights

        Note: Model target weight size should be 41645640 .
        weight file after converting to numpy should have size 20856776, therefore, it should be noted that the
        first 5 indexes are considered as the heading.
        """

        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, count=5, dtype=np.int32)
            buf = np.fromfile(fp, dtype=np.float32)
            start = 0

            for child in self.children():
                if isinstance(child, nn.Sequential):
                    for num_layer, layer in enumerate(child):
                        # Immersion in convolutional layers between residual
                        if isinstance(layer, nn.modules.conv.Conv2d):
                            if isinstance(child[num_layer + 1], nn.modules.BatchNorm2d):
                                conv_layer = child[num_layer]
                                batch_norm_layer = child[num_layer + 1]
                                start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)
                            else:
                                conv_layer = child[num_layer]
                                start = load_conv(buf, start, conv_layer)
                        # Immersion in residual blocks
                        elif isinstance(layer, ResBlock):
                            # Getting residual block
                            for residual_block in layer.children():
                                # Getting layers from residual block
                                for num_residual_layer, residual_layer in enumerate(residual_block.children()):
                                    if isinstance(residual_layer, nn.modules.conv.Conv2d):
                                        conv_layer = residual_block[num_residual_layer]
                                        batch_norm_layer = residual_block[num_residual_layer + 1]
                                        start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)
                
                # For weights with name "darknet53.conv.74"
                # These weights are available at https://pjreddie.com/media/files/darknet53.conv.74
                # if start == 40620640:
                #    break

            if start == buf.size:
                print("Darknet53 weight file upload successfully")
            else:
                print("Error: Darknet53 weight file upload unsuccessfully")
                exit(-1)
