"unet2d"
from torchsummary import summary
import torch
import torch.nn as nn
import hiddenlayer as hl


def init_weithts(net, init_type='normal', gain=0.02):
    "init weights"
    def init_func(mod):
        classname = mod.__class__.__name__
        if hasattr(mod, 'weight') and ((classname.find('conv') or classname.find('Linear'))):
            if init_type == 'normal':
                nn.init.normal_(mod.weight.data, 0.0, std=gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(mod.weight.data, gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(mod.weight.data)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(mod.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method {} not implemented'.format(init_type))
            if hasattr(mod, 'bias') and mod.bias is not None:
                nn.init.constant_(mod.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(mod.weight.data, 1.0, gain)
            nn.init.constant_(mod.bias.data, 0.0)
        print('initialize with {}'.format(init_type))

    net.apply(init_func)


class ConvBlock(nn.Module):
    "conv->bn->relu-conv-bn-relu"

    def __init__(self, c_in, c_out, bn=True):
        super(ConvBlock, self).__init__()
        if bn:
            self.func = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3,
                          stride=1, padding=1, bias=True),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True))
        else:
            self.func = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3,
                          padding=1, stride=1, bias=True),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.func(x)


class UpConv(nn.Module):
    "Upsampling"

    def __init__(self, c_in, c_out, up_scale=2):
        super(UpConv, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=up_scale,
                        align_corners=True, mode='bilinear'),
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    "Unet 2d"

    def __init__(self, c_in=1, c_out=1):
        super(Unet, self).__init__()
        self.conv1 = ConvBlock(c_in, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up5 = UpConv(512, 256)
        self.up_conv5 = ConvBlock(512, 256)
        self.up4 = UpConv(256, 128)
        self.up_conv4 = ConvBlock(256, 128)
        self.up3 = UpConv(128, 64)
        self.up_conv3 = ConvBlock(128, 64)
        self.up2 = UpConv(64, 32)
        self.up_conv2 = ConvBlock(64, 32)
        self.up_conv1 = nn.Conv2d(32, c_out, kernel_size=1)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.maxpool(x_1)
        x_2 = self.conv2(x_2)
        x_3 = self.maxpool(x_2)
        x_3 = self.conv3(x_3)
        x_4 = self.maxpool(x_3)
        x_4 = self.conv4(x_4)
        x_5 = self.maxpool(x_4)
        x_5 = self.conv5(x_5)

        up_x5 = self.up5(x_5)
        up_x5 = torch.cat((up_x5, x_4), dim=1)
        up_x5 = self.up_conv5(up_x5)
        up_x4 = self.up4(up_x5)
        up_x4 = torch.cat((up_x4, x_3), dim=1)
        up_x4 = self.up_conv4(up_x4)
        up_x3 = self.up3(up_x4)
        up_x3 = torch.cat((up_x3, x_2), dim=1)
        up_x3 = self.up_conv3(up_x3)
        up_x2 = self.up2(up_x3)
        up_x2 = torch.cat((up_x2, x_1), dim=1)
        up_x2 = self.up_conv2(up_x2)
        out = self.up_conv1(up_x2)
        out = torch.softmax(out,1)   #For crossentropy loss function
        # out = torch.log_softmax(out, 1)  # For BCELoss
        return out


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    UNET = Unet(c_in=1, c_out=1)
    INPUTS = torch.randn([1, 1, 512, 512])
    IM = hl.build_graph(UNET, INPUTS)
    # ave('unet')
    OUTPUTS = UNET(INPUTS)
    summary(UNET.cuda(), (1, 512, 512))
