import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .operation import FactorizedReduce, ReLUConvBN, OPS
from .genotype import  s3

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

class NASseg(nn.Module):
    "Search unet like network "

    def __init__(self, C, num_classes, genotype='unet', layers=12):
        super(NASseg, self).__init__()
        self.channel = C

        self.num_classes = num_classes
        self.layer = layers
        self.stem0 = nn.Sequential(nn.Conv2d(3, C//2, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(C//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(C//2, C, 3, stride=2,
                                             padding=1, bias=False),
                                   nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(
            C, C, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(C))

        c_pp, c_p, c_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//4, 2*layers//4, 3*layers//4]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, c_pp, c_p, c_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            c_pp, c_p = c_p, cell.multiplier*c_curr

        self.cells_decode = nn.ModuleList()
        reduction = False
        expansion_prev = False
        for i in range(layers):
            if i+1 in [layers//4, 2*layers//4, 3*layers//4]:
                expansion = True
                c_curr //= 2
            else:
                expansion = False

            cell = CellDecode(genotype, c_pp, c_p, c_curr,
                              expansion=expansion, expansion_prev=expansion_prev)
            expansion_prev = expansion
            self.cells_decode += [cell]
            c_pp, c_p = c_p, c_curr * cell.multiplier

        self.outputlayer = nn.Conv2d(64, num_classes, 1, 1)
        self._init_params()

    def forward(self, inputs):
        N, C, W, H = inputs.size()
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        middle_feature = []
        for _, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            middle_feature.append(s1)
        for i, cell in enumerate(self.cells_decode):
            s0, s1 = s1, cell(s0, s1 )
            if cell.expansion:
                # feature=F.interpolate(middle_feature[self.layer-i-1],scale_factor=2.,mode='bilinear',align_corners=True)
                feature = middle_feature[self.layer-i-1]
                s0 = s0+feature
        output = self.outputlayer(s1)
        output = F.interpolate(
            output, (W, H), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        return output

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)


if __name__ == "__main__":
    a = NASseg(16,3,s3,12)
    inputs=torch.randn(1,3,384,384)
    out=a(inputs)
    print(out.size())
