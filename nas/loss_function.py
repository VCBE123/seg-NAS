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


class Weight_DiceLoss(nn.Module):
    "Calculate the weighted dice loss"

    def __init__(self, smooth=1):
        super(Weight_DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        batch, classes, W, H = target.size()
        count = torch.full((batch, classes), W*H).cuda()
        weight = torch.sum(target.view(batch, classes, W*H), dim=-1)
        weight = count.div(weight)

        output = output.view(batch, classes, W*H)
        target = target.view(batch, classes, W*H)

        intersection = (output*target).sum(dim=-1)
        loss = 1-((2.*intersection+self.smooth)) / \
            (output.sum(dim=-1)+target.sum(dim=-1)+self.smooth)
        loss *= weight

        return loss.sum()
