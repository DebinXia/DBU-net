import tifffile
import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F


class UNet(nn.Module):
    """
    对图像对提取特征获得将moving图像进行形变的形变场。通过复杂的计算最终让网络计算出一个相对比较优秀的模型，直接预测出一个结果
    目前这个模型也可以适用于其他任务，只是一个接口
    初始化：
        配准：
            ndim：图像维度
            in_channel：输入通道，应该设置为2
            out_channel：输出通道，应该对应moving图像的维度，是moving的形变场
    """
    def conv_block(self, ndim, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        Conv = getattr(nn, 'Conv%dd' % ndim)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndim)
        block = torch.nn.Sequential(
            Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            BatchNorm(out_channels),
            torch.nn.ReLU(),
            Conv(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            BatchNorm(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, ndim, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        Conv = getattr(nn, 'Conv%dd' % ndim)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndim)
        ConvTranspose = getattr(nn, 'ConvTranspose%dd' % ndim)
        block = torch.nn.Sequential(
            Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            BatchNorm(mid_channel),
            torch.nn.ReLU(),
            Conv(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            BatchNorm(mid_channel),
            torch.nn.ReLU(),
            ConvTranspose(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def bottleneck_block(self, ndim, mid_channel):
        """"""
        Conv = getattr(nn, 'Conv%dd' % ndim)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndim)
        ConvTranspose = getattr(nn, 'ConvTranspose%dd' % ndim)
        block = torch.nn.Sequential(
                Conv(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                BatchNorm(mid_channel * 2),
                torch.nn.ReLU(),
                Conv(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
                BatchNorm(mid_channel),
                torch.nn.ReLU(),
                ConvTranspose(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2,
                              padding=1, output_padding=1),
                BatchNorm(mid_channel),
                torch.nn.ReLU(),
            )
        return block

    def final_block(self, ndim, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        Conv = getattr(nn, 'Conv%dd' % ndim)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndim)

        block = torch.nn.Sequential(
                    Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    BatchNorm(mid_channel),
                    torch.nn.ReLU(),
                    Conv(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    BatchNorm(out_channels),
                    torch.nn.ReLU()
                )
        return block

    def __init__(self, ndim, in_channel, out_channel):
        super(UNet, self).__init__()
        assert ndim in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndim
        self.ndim = ndim
        #Encode 32,32,64,64,128,96,64
        # 32, 64
        MaxPool = getattr(nn, 'MaxPool%dd' % ndim)
        """self.conv_encode1 = self.conv_block(ndim, in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = MaxPool(kernel_size=2)
        self.conv_encode2 = self.conv_block(ndim, 32, 32)  # 32, 64
        self.conv_maxpool2 = MaxPool(kernel_size=2)
        self.conv_encode3 = self.conv_block(ndim, 32, 64)  # 64, 128
        self.conv_maxpool3 = MaxPool(kernel_size=2)
        # Bottleneck
        mid_channel = 64  # 128
        self.bottleneck = self.bottleneck_block(ndim, mid_channel)
        # Decode
        self.conv_decode3 = self.expansive_block(ndim, 128, 64, 64)  # 256, 128, 64
        self.conv_decode2 = self.expansive_block(ndim, 96, 32, 32)  # 128, 64, 32

        # 少了一个反卷积过程
        self.final_layers = self.final_block(ndim, 64, 32, out_channel)  # 64, 32, out_channel"""
        self.conv_encode1 = self.conv_block(ndim, in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = MaxPool(kernel_size=2)
        self.conv_encode2 = self.conv_block(ndim, 32, 64)  # 32, 64
        self.conv_maxpool2 = MaxPool(kernel_size=2)
        self.conv_encode3 = self.conv_block(ndim, 64, 128)  # 64, 128
        self.conv_maxpool3 = MaxPool(kernel_size=2)
        # Bottleneck
        mid_channel = 128  # 128
        self.bottleneck = self.bottleneck_block(ndim, mid_channel)
        # Decode
        self.conv_decode3 = self.expansive_block(ndim, 256, 128, 64)  # 256, 128, 64
        self.conv_decode2 = self.expansive_block(ndim, 128, 64, 32)  # 128, 64, 32

        # 少了一个反卷积过程
        self.final_layers = self.final_block(ndim, 64, 32, out_channel)  # 64, 32, out_channel

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        残差连接
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        # print(encode_pool1.shape)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        # print(encode_pool2.shape)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # print(encode_pool3.shape)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # print(bottleneck1.shape)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        # print(cat_layer2.shape)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layers(decode_block1)
        return [final_layer]


if __name__ == '__main__':
    import numpy as np
    fixed = torch.tensor(np.zeros((1, 1, 64, 256, 256), dtype=np.float32)).to('cuda:1')
    moving = torch.tensor(np.zeros((1, 1, 64, 256, 256), dtype=np.float32)).to('cuda:1')
    model = UNet(ndim=3, in_channel=2, out_channel=3, )
    a = model(fixed, moving)
    print(a.Size())