"Construct the Unet like Searching network"
import torch
import torch.nn as nn
from torch.autograd import Variable
from .genotype import PRIMITIVES
from .operation import OPS, ReLUConvBN, FactorizedReduce


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
        return sum(w.cuda()*operation(x.cuda()) for w, operation in zip(weights, self.m_op))


class CellSearch(nn.Module):
    "Cell structure for searching"

    def __init__(self, steps, multiplier, c_pp, c_p, C, reduction, reduction_prev, switches):
        super(CellSearch, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_pp, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_pp, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(c_p, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                operation = MixedOp(C, stride, switch=switches[switch_count])
                self.cell_ops.append(operation)
                switch_count += 1

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            sum_ = sum(self.cell_ops[offset+i](h, weights[offset+j])
                       for j, h in enumerate(states))
            offset += len(states)
            states.append(sum_)
        return torch.cat(states[-self._multiplier:], dim=1)


class NASUnet(nn.Module):
    "Search unet like network "
    def __init__(self, C, num_classes, layers, criterion, steps, multiplier=4, stem_multiplier=3, switches_normal=None, switches_reduce=None):
        super(NASUnet, self).__init__()
        self.channel = C,
        self.num_classes = num_classes
        self.player = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier
        self.switch_normal = switches_normal
        switch_ons = []
        for i, _ in enumerate(switches_normal):
            ons = 0
            for j, _ in enumerate(switches_normal[i]):
                if switches_normal[i][j]:
                    ons += 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]

        c_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False), nn.BatchNorm2d(c_curr))
        self.stem0 = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(C),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(C, c_curr, 3, stride=2,
                                             padding=1, bias=False),
                                   nn.BatchNorm2d(c_curr))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(
            c_curr, c_curr, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(c_curr))

        c_pp, c_p, c_curr = c_curr, c_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                c_curr *= 2
                reduction = True
                cell = CellSearch(steps, multiplier, c_pp, c_p,
                                  c_curr, reduction, reduction_prev, switches_reduce)
            else:
                reduction = False
                cell = CellSearch(steps, multiplier, c_pp, c_p,
                                  c_curr, reduction, reduction_prev, switches_normal)
            reduction_prev = reduction
            self.cells += [cell]
            c_pp, c_p = c_p, multiplier*c_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self._initialize_alphas()

    def forward(self, inputs):
        pass

    def _initialize_alphas(self):
        "initialize arch-paramters with randn distribution"
        k = sum(1 for i in range(self.steps) for n in range(2+i))
        num_ops = self.switch_on
        self.alphas_normal = Variable(
            1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(
            1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        "return the arch-parameters"
        return self._arch_parameters
