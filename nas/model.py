import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .operation import FactorizedReduce, ReLUConvBN, OPS
from .genotype import  s3
from nas.Mix import mixnet_xl
from .RayNet import ASSP, SepConv
class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)  # 8
        self._steps = len(op_names) // 2  # 4
        self._concat = concat
        self.multiplier = len(concat)  # 4

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class CellDecode(nn.Module):
    "Cell structure for searching"

    def __init__(self, genotype, c_pp, c_p, C,  expansion=False, expansion_prev=False):
        super(CellDecode, self).__init__()
        self.expansion = expansion
        self.expansion_prev = expansion_prev
        self.preprocess0 = ReLUConvBN(c_pp, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(c_p, C, 1, 1, 0, affine=False)
        if expansion:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, expansion)

    def _compile(self, C, op_names, indices, concat, expansion):
        assert len(op_names) == len(indices)  # 8
        self._steps = len(op_names) // 2  # 4
        self._concat = concat
        self.multiplier = len(concat)  # 4

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride =1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices


    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        if self.expansion_prev:
            s0 = F.interpolate(s0, scale_factor=2.,
                               mode='bilinear', align_corners=True)
        s1 = self.preprocess1(s1)
        if self.expansion:
            s0 = F.interpolate(s0, scale_factor=2.,
                               mode='bilinear', align_corners=True)
            s1 = F.interpolate(s1, scale_factor=2.,
                               mode='bilinear', align_corners=True)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class NASRayNetEval(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEval, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=320, output_stride=16)
        self.decode_cell = CellDecode(genotype, 256, 128, 64, expansion_prev=True)

        self.low_cell1 = Cell(genotype, 48, 64, 16,reduction=False, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64,  64, 16,reduction=False, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)

        self.outcell1 = CellDecode( genotype,256, 64, 32,expansion=True, expansion_prev=True)
        self.cell1 = Cell(genotype, 48, 128, 32,reduction=False, reduction_prev=False)
        self.cell2 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)
        self.cell3 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)

        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample( scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-1])

        decode1 = self.decode_cell(aspp, middle_feature[-2])


        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])
        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat3=self.low_cell3(low_feat1,low_feat2)
        low_feat4=self.low_cell4(low_feat2,low_feat3)
        out1 = self.outcell1(decode1, low_feat4)
        out2 = self.cell1(middle_feature[0],out1)
        out3= self.cell2(out1,out2)
        out4= self.cell3(out2,out3)
        out = self.out(out4)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out


class NASRayNetEvalDense(nn.Module):
    "adopt from raynet_v0"

    def __init__(self, pretrained=True, num_classes=3, genotype='ray1',layer=12):
        super(NASRayNetEvalDense, self).__init__()
        self.encode = mixnet_xl(pretrained=pretrained,
                                num_classes=num_classes,head_conv=None)    # 48-96-96 64-48-48 128-24-24 320-12-12
        self.aspp = ASSP(in_channels=320, output_stride=16)
        self.decode_cell = CellDecode(genotype, 256, 128, 64, expansion_prev=True)

        self.low_cell1 = Cell(genotype, 48, 64, 16,reduction=False, reduction_prev=True)
        self.low_cell2 = Cell(genotype, 64,  64, 16,reduction=False, reduction_prev=False)
        self.low_cell3 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)
        self.low_cell4 = Cell(genotype, 64, 64, 16,reduction=False, reduction_prev=False)

        self.outcell1 = CellDecode( genotype,256, 64, 32,expansion=True, expansion_prev=True)
        self.cell1 = Cell(genotype, 48, 128, 32,reduction=False, reduction_prev=False)
        self.cell2 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)
        self.cell3 = Cell(genotype, 128, 128, 32,reduction=False, reduction_prev=False)

        self.out = SepConv(128, num_classes, 1, 1, 0)
        self.up4 = nn.Upsample( scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        _, middle_feature = self.encode.forward_features(inputs)
        aspp = self.aspp(middle_feature[-1])

        decode1 = self.decode_cell(aspp, middle_feature[-2])


        low_feat1 = self.low_cell1( middle_feature[0], middle_feature[1])

        low_feat2=self.low_cell2(middle_feature[1],low_feat1)
        low_feat3=self.low_cell3(low_feat1,low_feat2)
        low_feat4=self.low_cell4(low_feat2,low_feat3)
        out1 = self.outcell1(decode1, low_feat4)
        out2 = self.cell1(middle_feature[0],out1)
        out3= self.cell2(out1,out2)
        out4= self.cell3(out2,out3)
        out = self.out(out4)
        out = self.up4(out)
        out = torch.softmax(out, 1)
        return out

if __name__ == "__main__":
    a = NASRayNetEval(16,3,s3,12)
    inputs=torch.randn(2,3,384,384)
    out=a(inputs)
    print(out.size())
