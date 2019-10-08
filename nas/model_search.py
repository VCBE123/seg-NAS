"Construct the Unet like Searching network"
import torch.nn.functional as F
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
        for i in enumerate(switch):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False)

                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self.m_op.append(op)

    def forward(self, x, weights):
        return sum(w.cuda()*op(x.cuda()) for w, op in zip(weights, self.m_op))


class Cell_Search(nn.Module):
    "Cell structure for searching"
    def __init__(self, steps, multiplier, C_pp, C_p, C, reduction, reduction_prev, switches):
        super(Cell_Search, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_pp, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_p, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count])
                self.cell_ops.append(op)
                switch_count += 1

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+i](h, weights[offset+j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)
