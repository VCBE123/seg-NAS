
"train progress"
import os
import time
import argparse
import cv2
import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from nas import NASRayNetEval_aspp_base, ray2, RayNet_v0, ray3, RayNet_v1,NASRayNetEval_aspp_2

from dataloader import get_follicle
from utils import AverageMeter, get_dice_ovary, get_dice_follicle, get_hd
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='low')
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='0')
    return parser.parse_args()


ARGS = get_parser()
os.environ['cuda_visible_devices'] = ARGS.gpus


def main():
    "evaluate main"
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpus
    np.random.seed(ARGS.seed)
    cudnn.benchmark = True
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)

    model =NASRayNetEval_v3(genotype=ray2)
    model = nn.DataParallel(model)
    model = model.cuda()

    state_dict = torch.load('exp2/train--nasray_ray2_aspp_4cell_Dense-191118-Nov111574057702/model_best.pth.tar')
    model.load_state_dict(state_dict['state_dict'])
    model = model.module

    _, val_loader = get_follicle(1,1,return_path=True)
    epoch_start = time.time()
    valid_dice_follicle, valid_dice_ovary, valid_hd = infer(val_loader, model)
    epoch_duration = time.time()-epoch_start
    print("valid_dice_follicle:{} vald_dice_ovary:{} valid_hd:{} duration: {}s".format(
        valid_dice_follicle, valid_dice_ovary, valid_hd, epoch_duration))


def infer(valid_loader, model):
    "validate func"
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    hd_meter = AverageMeter()
    # for m in model.modules():
    # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    # m.track_running_stats = False

    model.eval()
    print(model.training)
    count = 0
    for inputs, targets,image_path,label_path in tqdm.tqdm(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
        dice_follicle = get_dice_follicle(logits, targets)
        dice_ovary = get_dice_ovary(logits, targets)

        # hd = get_hd(logits, targets)
        save_mask= True
        if save_mask:
            pred = logits.cpu().numpy()
            segmap = np.argmax(pred.squeeze(), axis=0)
            segmap[segmap == 1] = 128
            segmap[segmap == 2] = 255
            input_image=cv2.imread(image_path[0])
            # input_image=np.resize(input_image,[384,384])
            target_image=cv2.imread(label_path[0])
            target_image[target_image==1]=128
            target_image[target_image==2]=255
            # target_image=np.resize(target_image,[384,384])
            # segmap=np.concatenate([input_image[:,:,0],target_image[:,:,0],segmap],1)
            segmap=np.concatenate([input_image[:,:,0],segmap],1)
            cv2.imwrite('logs/base_line/{}.png'.format(count), segmap)
        count += 1
        batch_size = inputs.size(0)
        dice_follicle_meter.update(dice_follicle, batch_size)
        dice_ovary_meter.update(dice_ovary, batch_size)
        hd_meter.update(0, batch_size)

    return dice_follicle_meter.avg, dice_ovary_meter.avg, hd_meter.avg


if __name__ == '__main__':
    main()
