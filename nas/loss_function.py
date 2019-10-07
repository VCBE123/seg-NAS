"dice loss"
# import torch


def dice_loss(inputs, target):
    "caculate dice loss"
    smooth = 1.
    iflaten = inputs.view(-1)
    tflaten = target.view(-1)
    intersection = (iflaten*tflaten).sum()
    return 1-((2.*intersection+smooth)/(iflaten.sum()+tflaten.sum()+smooth))
