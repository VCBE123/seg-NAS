import numpy as np

class bestMeter(object):
	"""
	record the best
	"""

	def __init__(self, best=-1, mode='max'):
		self.best = best
		self.mode = mode

	def update(self, val):
		if self.mode == 'max':
			if val > self.best:
				self.best = val
				return True
			else:
				return False
		elif self.mode == 'min':
			if val < self.best:
				self.best = val
				return True
			else:
				return False


class averageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def get_dice(pred, mask, threshold=0.5):
	'''
	calculate dice coefficient
	:param pred:
	:param mask:
	:param threshold:
	:return:
	'''
	pred = (pred > threshold).astype(int)
	iflaten = pred.flatten()
	tflaten = mask.flatten()
	intersection = (iflaten * tflaten).sum()
	return (2. * intersection) / (iflaten.sum() + tflaten.sum() + 1e-6)
