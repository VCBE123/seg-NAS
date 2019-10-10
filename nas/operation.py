"offer operations"
import torch
import torch.nn as nn
import numpy as np

__all__ = ['OPS', 'ReLUConvBN', 'FactorizedReduce']
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'md_conv_35': lambda C, stride, affine: MDConv(C, stride, n_chunks=2),
    'md_conv_357': lambda C, stride, affine: MDConv(C, stride, n_chunks=3),
    'se': lambda C, stride, affine: SE(C, C, stride),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):
    "Relu->conv->Bn block"

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.ops = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.ops(x)


class DilConv(nn.Module):
    "dilation conv"

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.ops = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.ops(x)


class SepConv(nn.Module):
    "sepwise conv"

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.ops = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.ops(x)


def split_layer(total_channels, num_groups):
    "split channels to groups"
    split = [int(np.ceil(total_channels / num_groups))
             for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthWiseConv2d(nn.Module):
    """
    Depthwise conv
    """

    def __init__(self, in_channels, kernel_size, stride, bias=False):
        super(DepthWiseConv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                        stride=stride,
                                        groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class MDConv(nn.Module):
    """
    mixconv
    """

    def __init__(self, C_out, stride, n_chunks=3):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channel = split_layer(C_out, n_chunks)
        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(
                DepthWiseConv2d(self.split_out_channel[idx], kernel_size=kernel_size, stride=stride, bias=False))

    def forward(self, x):
        split = torch.split(x, self.split_out_channel, dim=1)
        out = torch.cat([layer(s)
                         for layer, s in zip(self.layers, split)], dim=1)
        return out


class Identity(nn.Module):
    "Identity"

    # def __init__(self):
    # super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    "none operation"

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        batches, channels, heights, widths = x.size()
        heights //= self.stride
        widths //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(
                    batches, channels, heights, widths).fill_(0)
        else:
            padding = torch.FloatTensor(
                batches, channels, heights, widths).fill_(0)
        return padding


class FactorizedReduce(nn.Module):
    "FactorizedReduce"

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.bn2d = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn2d(out)
        return out


class SE(nn.Module):
    """
    se block
    """

    def __init__(self, C_in, C_out, stride):
        super(SE, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.gate = F.sigmoid   #comment due to the complain
        self.reduce = C_in
        self.conv_reduce = nn.Conv2d(C_in, C_in, 1, bias=False)
        self.conv_expand = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.stride = stride
        if self.stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.stride == 2:
            x = self.maxpool(x)
        x_se = x.view(x.size(0), x.size(1), -
                      1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv_expand(x_se)
        x = x * torch.sigmoid(x_se)
        return x
