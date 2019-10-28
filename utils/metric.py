" avergeMeter,bestMeter,dice"
import numpy as np
import hausdorff as hd
import cv2


class BestMeter(object):
    """
    record the best
    """

    def __init__(self, best=-1, mode='max'):
        self.best = best
        self.mode = mode

    def update(self, val):
        "update best"
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.reset()

    def reset(self):
        "reset to 0."
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        "update avg, sum "
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


def get_dice_follicle(pred, mask):
    '''
    calculate dice coefficient of follicle
    :param pred: tensor
    :param mask:  tensor
    :param threshold:
    :return:
    '''
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)
    pred[pred == 2] = 0
    mask = mask.cpu().numpy()
    mask = np.argmax(mask, 1)
    mask[mask == 2] = 0
    dice = 0.
    for i in range(mask.shape[0]):
        iflaten = pred[i, ...].flatten()
        tflaten = mask[i, ...].flatten()
        intersection = (iflaten * tflaten).sum()
        dice += (2. * intersection) / (iflaten.sum() + tflaten.sum() + 1e-6)
    return dice/mask.shape[0]


def get_dice_ovary(pred, mask):
    '''
    calculate dice coefficient of ovary
    :param pred: tensor
    :param mask:  tensor
    :param threshold:
    :return:
    '''
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)
    pred[pred == 2] = 1
    mask = mask.cpu().numpy()
    mask = np.argmax(mask, 1)
    mask[mask == 2] = 1
    dice = 0.
    for i in range(mask.shape[0]):
        iflaten = pred[i, ...].flatten()
        tflaten = mask[i, ...].flatten()
        intersection = (iflaten * tflaten).sum()
        dice += (2. * intersection) / (iflaten.sum() + tflaten.sum() + 1e-6)
    return dice/mask.shape[0]


def get_hd(pred, mask):
    "calculate the hausdorff distance (in euclidean space)  in between follicle contours"
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)
    pred[pred == 2] = 0
    mask = mask.cpu().numpy()
    mask = np.argmax(mask, 1)
    mask[mask == 2] = 0
    for i in range(mask.shape[0]):
        predi=pred[i,...].copy()
        maski=mask[i,...].copy()
        _,cp,_=cv2.findContours(predi,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        _,cm,_=cv2.findContours(maski,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        hd=cv2.createHausdorffDistanceExtractor()
        sd=cv2.createShapeContextDistanceExtractor()
        d1=hd.computeDistance(cp[0],cm[0])
        d2=sd.computeDistance(cp[0],cm[0])
        print(d1)
        print(d2)
    return d1,d2


