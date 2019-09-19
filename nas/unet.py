import torch
from torchsummary import summary
import torch.nn as nn
from torch.nn import init
import hiddenlayer as hl

def init_weithts(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and ((classname.find('conv') or classname.find('Linear'))):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, std=gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, std=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data)
			elif init_type == "orthogonal":
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method {} not implemented'.format(init_type))
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d')!=-1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
		print('initialize with {}'.format(init_type))

	net.apply(init_func)


class Conv_block(nn.Module):
	def __init__(self, c_in, c_out, bn=True):
		super(Conv_block, self).__init__()
		if bn:
			self.func = nn.Sequential(
				nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,padding=1, bias=True),
				nn.BatchNorm2d(c_out),
				nn.ReLU(inplace=True),
				nn.Conv2d(c_out, c_out, kernel_size=3,padding=1, bias=True),
				nn.BatchNorm2d(c_out),
				nn.ReLU(inplace=True))
		else:
			self.func = nn.Sequential(
				nn.Conv2d(c_in, c_out, kernel_size=3,padding=1, stride=1, bias=True),
				nn.BatchNorm2d(c_out),
				nn.ReLU(inplace=True),
				nn.Conv2d(c_out, c_out, kernel_size=3,padding=1, bias=True),
				nn.ReLU(inplace=True))

	def forward(self, x):
		return self.func(x)


class Up_conv(nn.Module):
	def __init__(self, c_in, c_out, up_scale=2):
		super(Up_conv, self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=up_scale,align_corners=True,mode='bilinear'),
			nn.Conv2d(c_in, c_out, kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

	def forward(self, x):
		return self.up(x)


class Unet(nn.Module):
	def __init__(self, c_in=1, c_out=1):
		super(Unet, self).__init__()
		self.conv1 = Conv_block(c_in, 32)
		self.conv2 = Conv_block(32, 64)
		self.conv3 = Conv_block(64, 128)
		self.conv4 = Conv_block(128, 256)
		self.conv5 = Conv_block(256, 512)

		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.up5 = Up_conv(512, 256)
		self.up_conv5 = Conv_block(512, 256)
		self.up4 = Up_conv(256, 128)
		self.up_conv4 = Conv_block(256, 128)
		self.up3 = Up_conv(128, 64)
		self.up_conv3 = Conv_block(128, 64)
		self.up2 = Up_conv(64, 32)
		self.up_conv2 = Conv_block(64, 32)
		self.up_conv1 = nn.Conv2d(32, c_out, kernel_size=1)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.maxpool(x1)
		x2 = self.conv2(x2)
		x3 = self.maxpool(x2)
		x3 = self.conv3(x3)
		x4 = self.maxpool(x3)
		x4 = self.conv4(x4)
		x5 = self.maxpool(x4)
		x5 = self.conv5(x5)

		up_x5 = self.up5(x5)
		up_x5 = torch.cat((up_x5, x4), dim=1)
		up_x5 = self.up_conv5(up_x5)
		up_x4 = self.up4(up_x5)
		up_x4 = torch.cat((up_x4, x3), dim=1)
		up_x4 = self.up_conv4(up_x4)
		up_x3 = self.up3(up_x4)
		up_x3 = torch.cat((up_x3, x2), dim=1)
		up_x3 = self.up_conv3(up_x3)
		up_x2 = self.up2(up_x3)
		up_x2 = torch.cat((up_x2, x1), dim=1)
		up_x2 = self.up_conv2(up_x2)
		out = self.up_conv1(up_x2)
		out=torch.sigmoid(out)
		return out


if __name__ == '__main__':
	import os
	os.environ['CUDA_VISIBLE_DEVICES']='2'
	net = Unet(c_in=1, c_out=1)
	input = torch.randn([1,1, 512, 512])
	# im=hl.build_graph(net,input)
	# im.save('unet')
	out=net(input)
	summary(net.cuda(), (1, 512, 512))