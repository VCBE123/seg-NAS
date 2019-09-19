import torch

def dice_loss(input,target):
	smooth=1.
	iflaten=input.view(-1)
	tflaten=target.view(-1)
	intersection=(iflaten*tflaten).sum()
	return 1-((2.*intersection+smooth)/(iflaten.sum()+tflaten.sum()+smooth))