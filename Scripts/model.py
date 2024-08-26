# model.py
from util import *
import torch
import torch.nn as nn
import torchvision


class Atrous_Convolution(nn.Module):
    def __init__(self, img_size=64, in_channels=32, out_channels=64, dilation_rate=0, kernel_size=3, padding='same'):
        super().__init__()
        self.img_size = img_size
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels // 2,
                      kernel_size=self.kernel_size[i],
                      padding=self.padding,
                      dilation=self.dilation_rate[i]) for i in range(len(dilation_rate))
        ])
        self.batchnorm = nn.BatchNorm2d(self.in_channels * 2 + self.in_channels)
        self.batchnorm_1 = nn.BatchNorm2d(self.out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.one_cross_one_conv = nn.Conv2d(in_channels=self.out_channels * 2 + self.in_channels,
                                            out_channels=self.out_channels, kernel_size=1, dilation=1, padding='same')

    def forward(self, x):
        op = [block(x) for block in self.conv]
        op.append(self.maxpool(x))
        op = torch.cat(op, axis=1)
        op = self.batchnorm(op)
        op = self.one_cross_one_conv(op)
        op = self.batchnorm_1(op)
        op = self.relu(op)
        return op


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        atrous_convolutions = [
            Atrous_Convolution(in_channels=1024, out_channels=1024, dilation_rate=[1, 6, 12, 18], kernel_size=[1, 3, 3, 3]),
            Atrous_Convolution(in_channels=512, out_channels=512, dilation_rate=[1, 6, 12, 18], kernel_size=[1, 3, 3, 3]),
            Atrous_Convolution(in_channels=256, out_channels=256, dilation_rate=[1, 6, 12, 18], kernel_size=[1, 3, 3, 3]),
            Atrous_Convolution(in_channels=64, out_channels=64, dilation_rate=[1, 6, 12, 18], kernel_size=[1, 3, 3, 3])
        ]

        self.atrous_convolutions = nn.ModuleList(atrous_convolutions)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            if i < 5:
                x = block(x, self.atrous_convolutions[i - 1](pre_pools[key]))
            else:
                x = block(x, pre_pools[key])

        output_feature_map = x
        x = self.out(x)
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

def get_model():
    model = UNetWithResnet50Encoder()
    return model