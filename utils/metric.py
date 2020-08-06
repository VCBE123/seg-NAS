" avergeMeter,bestMeter,dice"
import numpy as np
import cv2
import surface_distance as surfdist
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




def asd(pred,mask):
    """
    calculate average surface distance
    """
    surface_distances = surfdist.compute_surface_distances(
    mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    return avg_surf_dist

def hd95(pred,mask):
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)
    pred_ovary=pred>1
    pred_follicle=pred==1

    mask = mask.cpu().numpy()
    mask = np.argmax(mask, 1)
    mask_ovary=mask>1
    mask_follicle=mask==1


    surface_distances_ovary = surfdist.compute_surface_distances( mask_ovary, pred_ovary, spacing_mm=(1.0, 1.0, 1.0))
    hd_dist_95_ovary = surfdist.compute_robust_hausdorff(surface_distances_ovary, 95)
    asd_ovary=surfdist.compute_average_surface_distance(surface_distances_ovary)
    



    surface_distances_follicle = surfdist.compute_surface_distances( mask_follicle, pred_follicle, spacing_mm=(1.0, 1.0, 1.0))
    hd_dist_95_follicle = surfdist.compute_robust_hausdorff(surface_distances_follicle, 95)
    asd_follicle=surfdist.compute_average_surface_distance(surface_distances_follicle)

    return hd_dist_95_ovary,hd_dist_95_follicle, np.array(asd_ovary).mean(), np.array(asd_follicle).mean()
    
def mean_IU(pred,mask):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_



def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

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
    jc=0
    for i in range(mask.shape[0]):
        iflaten = pred[i, ...].flatten()
        tflaten = mask[i, ...].flatten()
        intersection = (iflaten * tflaten).sum()
        dice += (2. * intersection) / (iflaten.sum() + tflaten.sum() + 1e-6)
        
        jc+=intersection/(iflaten.sum()+tflaten.sum()-intersection+1e-6)


    return dice/mask.shape[0],jc/mask.shape[0]


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
    jc=0
    for i in range(mask.shape[0]):
        iflaten = pred[i, ...].flatten()
        tflaten = mask[i, ...].flatten()
        intersection = (iflaten * tflaten).sum()
        dice += (2. * intersection) / (iflaten.sum() + tflaten.sum() + 1e-6)

        jc+=intersection/( iflaten.sum()+tflaten.sum()-intersection+1e-6)
    return dice/mask.shape[0], jc/mask.shape[0]


def get_hd(pred, mask):
    "calculate the hausdorff distance (in euclidean space)  in between follicle contours"
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)
    pred[pred == 2] = 0
    mask = mask.cpu().numpy()
    mask = np.argmax(mask, 1)
    mask[mask == 2] = 0
    hd = cv2.createHausdorffDistanceExtractor()
    for i in range(mask.shape[0]):
        _, predi = cv2.threshold(pred[i, ...].copy().astype(np.uint8), 0, 1, 0)
        _, maski = cv2.threshold(mask[i, ...].copy().astype(np.uint8), 0, 1, 0)
        cp,_  = cv2.findContours(predi.astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cm,_  = cv2.findContours(maski.astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cps=np.concatenate(cp,axis=0)
        cms=np.concatenate(cm,axis=0)
    d1 = hd.computeDistance(cps, cms)
    return d1
