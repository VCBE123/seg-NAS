import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .genotype import PRIMITIVES
from .operation import OPS, ReLUConvBN, FactorizedReduce
from nas.Mix import mixnet_xl
from .model_search import CellSearch, CellDecode


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


class MixedOp(nn.Module):
    "mix all operation"

    def __init__(self, C, stride, switch):
        super(MixedOp, self).__init__()
        self.m_op = nn.ModuleList()
        for i, swi in enumerate(switch):
            if swi:
                primitive = PRIMITIVES[i]
                operation = OPS[primitive](C, stride, False)

                if 'pool' in primitive:
                    operation = nn.Sequential(
                        operation, nn.BatchNorm2d(C, affine=False))
                self.m_op.append(operation)

    def forward(self, x, weights):
        # Fixme debug on cpu
        return sum(w*operation(x) for w, operation in zip(weights, self.m_op))


# class CellNormal(nn.Module):
#     " Normal Cell structure for searching"

#     def __init__(self, c_pp, c_p, C, switches, reduce_prev=True):
#         super(CellNormal, self).__init__()
#         self.reduce_prev = reduce_prev
#         if self.reduce_prev:
#             self.preprocess0 = FactorizedReduce(c_pp, C, affine=False)
#         else:
#             self.preprocess0 = ReLUConvBN(c_pp, C, 1, 1, 0, affine=False)

#         self.preprocess1 = ReLUConvBN(c_p, C, 1, 1, 0, affine=False)
#         self._steps = 4
#         self._multiplier = 3
#         self.cell_ops = nn.ModuleList()
#         switch_count = 0
#         for i in range(self._steps):
#             for j in range(2+i):
#                 stride = 1
#                 operation = MixedOp(C, stride, switch=switches[switch_count])
#                 self.cell_ops.append(operation)
#                 switch_count += 1

#     def forward(self, s0, s1, weights):
#         s0 = self.preprocess0(s0)
#         s1 = self.preprocess1(s1)
#         states = [s0, s1]
#         offset = 0
#         for _ in range(self._steps):
#             sum_ = sum(self.cell_ops[offset+j](h, weights[offset+j])
#                        for j, h in enumerate(states))
#             offset += len(states)
#             states.append(sum_)
#         return torch.cat(states[-self._multiplier:], dim=1)


# class CellDecode(nn.Module):
#     "Decoder Cell structure for searching"

#     def __init__(self, CPP, CP, C, switches, expan_scale=2):
#         super(CellDecode, self).__init__()
#         self._steps = 4
#         self._multiplier = 4
#         self.expan_scale = expan_scale
#         self.preprocess = ReLUConvBN(CP, C, 3, 1, 1)
#         self.cell_ops = nn.ModuleList()
#         switch_count = 0
#         for i in range(self._steps):
#             for j in range(2+i):
#                 stride = 1
#                 operation = MixedOp(C, stride, switch=switches[switch_count])
#                 self.cell_ops.append(operation)
#                 switch_count += 1

#     def forward(self, s0, weights):
#         if self.expan_scale > 1:
#             s0 = F.interpolate(s0, scale_factor=self.expan_scale,
#                                mode='bilinear', align_corners=True)
#         s0 = self.preprocess(s0)
#         states = [s0, s0]
#         offset = 0
#         for _ in range(self._steps):
#             sum_ = sum(self.cell_ops[offset+j](h, weights[offset+j])
#                        for j, h in enumerate(states))
#             offset += len(states)
#             states.append(sum_)
#         return torch.cat(states[-self._multiplier:], dim=1)


class NASRayNet(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, switches_normal=None, switches_expansion=None):
        super(NASRayNet, self).__init__()
        self.steps = 4
        switch_ons = []
        for i, _ in enumerate(switches_normal):
            ons = 0
            for j, _ in enumerate(switches_normal[i]):
                if switches_normal[i][j]:
                    ons += 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes, head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=192, output_stride=16)
        self.decode_cell1 = CellDecode(4,4, 192, 256, 64, switches=switches_expansion)
        self.decode_cell2 = CellDecode(4,4, 128, 256, 64, switches=switches_expansion,expansion_prev=True)

        self.low_cell = CellSearch(4,4,48, 64, 16,True, switches_normal)

        self.outcell1 = CellDecode(4,4, 64, 256, 32,switches_expansion, expansion_prev=True)
        self.outcell2= CellSearch(4,4,192,168,32,False,switches_normal)
        self.outcell3= CellSearch(4,4,192,168,32,False,switches_normal)

        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self._initialize_alphas()

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-2])            # 128-24-24
        # aspp = middle_feature[-1]

        weights = F.softmax(self.alphas_expansion, dim=-1)
        decode1 = self.decode_cell1(middle_feature[-2],aspp, weights)
        decode2 = self.decode_cell2(middle_feature[-3],decode1, weights)

        weights = F.softmax(self.alphas_normal, dim=-1)
        low_feat1 = self.low_cell( middle_feature[0], middle_feature[1], weights)

        weights = F.softmax(self.alphas_expansion, dim=-1)

        out = self.outcell1(low_feat1, decode2, weights)
        # out = self.outcell2(out,middle_feature[0])
        # out = self.outcell3(out,middle_feature[0])

        out = self.out(out)
        out = self.up2(out)
        out = torch.softmax(out, 1)
        return out

    def _initialize_alphas(self):
        "initialize arch-paramters with randn distribution"
        k = sum(1 for i in range(self.steps) for n in range(2+i))
        num_ops = self.switch_on
        self.alphas_normal = Variable(
            1e-3*torch.randn(k, num_ops), requires_grad=True)  # Fixme debeg on gpu
        self.alphas_expansion = Variable(
            1e-3*torch.randn(k, num_ops), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_expansion,
        ]

    def arch_parameters(self):
        "return the arch-parameters"
        return self._arch_parameters


class NASRayNet_v1(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, switches_normal=None, switches_expansion=None):
        super(NASRayNet_v1, self).__init__()
        self.steps = 4
        switch_ons = []
        for i, _ in enumerate(switches_normal):
            ons = 0
            for j, _ in enumerate(switches_normal[i]):
                if switches_normal[i][j]:
                    ons += 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]
        # 48-96-96 64-48-48 128-24-24 320-12-12
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes, head_conv=None)
        self.aspp = ASSP(in_channels=320, output_stride=8)
        # self.reduce=nn.Conv2d(1536,128,1,1)
        self.decode_cell = CellDecode(
            256, 64, switches=switches_expansion, expan_scale=2)
        self.decode_cell_1 = CellDecode(
            256, 64, switches=switches_expansion, expan_scale=2)
        self.decode_cell_2 = CellDecode(
            256, 64, switches=switches_expansion, expan_scale=2)
        self.low_cell = CellNormal(
            48, 48, 64, switches_normal, reduce_prev=False)

        self.outcell1 = CellDecode(448, 128, switches_expansion, expan_scale=1)
        self.outcell2 = CellDecode(512, 128, switches_expansion, expan_scale=1)

        self.out = SepConv(512, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self._initialize_alphas()

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-1])

        weights = F.softmax(self.alphas_expansion, dim=-1)
        decode = self.decode_cell(aspp, weights)
        decode = self.decode_cell_1(decode, weights)
        decode = self.decode_cell_2(decode, weights)

        weights = F.softmax(self.alphas_normal, dim=-1)
        low_feat1 = self.low_cell(
            middle_feature[0], middle_feature[0], weights)

        weights = F.softmax(self.alphas_expansion, dim=-1)

        out = torch.cat([low_feat1, decode], dim=1)

        out = self.outcell1(out, weights)
        out = self.outcell2(out, weights)
        out = self.out(out)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out

    def _initialize_alphas(self):
        "initialize arch-paramters with randn distribution"
        k = sum(1 for i in range(self.steps) for n in range(2+i))
        num_ops = self.switch_on
        self.alphas_normal = Variable(
            1e-3*torch.randn(k, num_ops), requires_grad=True)  # Fixme debeg on gpu
        self.alphas_expansion = Variable(
            1e-3*torch.randn(k, num_ops), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_expansion,
        ]

    def arch_parameters(self):
        "return the arch-parameters"
        return self._arch_parameters


if __name__ == '__main__':

    import copy
    switches = []
    for i in range(14):
        switches.append([True for _, _ in enumerate(PRIMITIVES)])
    switches_normal = copy.deepcopy(switches)
    switches_expansion = copy.deepcopy(switches)
    criterion = nn.BCELoss()
    net = NASRayNet(switches_normal=switches_normal,
                       switches_expansion=switches_expansion)

    inputs = torch.randn([2, 3, 384, 384])
    output = net(inputs)
    print(output.size())
