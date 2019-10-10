"dice loss"
# import torch
import torch
import torch.nn as nn


def dice_loss(inputs, target):
    "caculate dice loss"
    smooth = 1.
    iflaten = inputs.view(-1)
    tflaten = target.view(-1)
    intersection = (iflaten*tflaten).sum()
    return 1-((2.*intersection+smooth)/(iflaten.sum()+tflaten.sum()+smooth))


class WeightDiceLoss(nn.Module):
    "Calculate the weighted dice loss"

    def __init__(self, smooth=1, weight=False):
        super(WeightDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, output, target):
        batch, classes, width, height = target.size()
        if self.weight:
            count = torch.full((batch, classes), width*height).cuda()
            weight = torch.sum(target.view(
                batch, classes, width*height), dim=-1)
            weight = count.div(weight)

        output = output.view(batch, classes, width*height)
        target = target.view(batch, classes, width*height)

        intersection = (output*target).sum(dim=-1)
        loss = 1-((2.*intersection+self.smooth)) / \
            (output.sum(dim=-1)+target.sum(dim=-1)+self.smooth)
        if self.weight:
            loss *= weight

        return loss.sum()
