import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from Mix import mixnet_xl


def initialize_weights(*nnmodels):
    "initial with kaiming"
    for model in nnmodels:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    "ASPP Atrous Spatial Pyramid Pooling"

    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [
            8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 32]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=x.size()[
                           2:], mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


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


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps)
        if self.activate:
            assert (activation is not None)
            if isfunction(activation):
                self.activ = activation()
            elif isinstance(activation, str):
                if activation == "relu":
                    self.activ = nn.ReLU(inplace=True)
                elif activation == "relu6":
                    self.activ = nn.ReLU6(inplace=True)
                else:
                    raise NotImplementedError()
            else:
                self.activ = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class RayNet(nn.Module):
    "adopt from gao ray"

    def __init__(self, pretrained=True, num_classes=3):
        super(RayNet, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes, head_conv=None)
        self.aspp = ASSP(in_channels=320, output_stride=8)
        self.low_conv = SepConv(48, 256, 1, 1, 0)
        self.up8 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.outconv1 = SepConv(512, 512, 3, 1, 1)
        self.outconv2 = SepConv(512, 512, 3, 1, 1)
        self.out = SepConv(512, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode(inputs)

        low_feat = self.low_conv(middle_feature[0])

        aspp_out = self.aspp(middle_feature[-1])
        up_aspp = self.up8(aspp_out)

        cat = torch.cat([low_feat, up_aspp], dim=1)
        out = self.outconv1(cat)
        out = self.outconv2(out)
        out = self.out(out)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out


class RayNet_v0(nn.Module):
    "adopt from gao ray"

    def __init__(self, encode='mixnet_xl', pretrained=True, num_classes=3):
        super(RayNet_v0, self).__init__()
        self.encode = timm.create_model(
            encode, pretrained=pretrained, num_classes=num_classes)
        self.aspp = ASSP(in_channels=1536, output_stride=8)
        self.low_conv = ConvBlock(48, 256, 1, 1, 0)
        self.up8 = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.outconv1 = ConvBlock(512, 512, 3, 1, 1)
        self.outconv2 = ConvBlock(512, 512, 3, 1, 1)
        self.out = ConvBlock(512, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode(inputs)

        aspp_out = self.aspp(middle_feature[1])
        low_feat = self.low_conv(middle_feature[0])
        up_aspp = self.up8(aspp_out)

        cat = torch.cat([low_feat, up_aspp], dim=1)
        out = self.outconv1(cat)
        out = self.outconv2(out)
        out = self.out(out)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out


if __name__ == '__main__':
    # m = RayNet()
    # inputs = torch.randn([2, 3, 416, 416])
    # out = m(inputs)
    # print(out.size())
    pass
