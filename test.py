
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
from nas import RayNet
from dataloader import get_follicle
from utils import AverageMeter, get_dice_ovary, get_dice_follicle
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='unet')
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='6')
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

    model = RayNet()
    model = nn.DataParallel(model)
    model = model.cuda()
    state_dict = torch.load('model_best.pth.tar')
    model.load_state_dict(state_dict['state_dict'])

    _, val_loader = get_follicle(ARGS)
    epoch_start = time.time()
    valid_dice_follicle, valid_dice_ovary = infer(val_loader, model)
    epoch_duration = time.time()-epoch_start
    print("valid_dice_follicle:{} vald_dice_ovary:{} duration: {}s".format(
        valid_dice_follicle, valid_dice_ovary, epoch_duration))


def infer(valid_loader, model):
    "validate func"
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    # model.eval()
    count = 0
    for inputs, targets in tqdm.tqdm(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)

        dice_follicle = get_dice_follicle(logits, targets)
        dice_ovary = get_dice_ovary(logits, targets)
        # pred = logits.cpu().numpy()
        # segmap = np.argmax(pred.squeeze(), axis=0)
        # segmap[segmap == 1] = 128
        # segmap[segmap == 2] = 255

        # cv2.imwrite('logs/fig/{}.png'.format(count), segmap)
        count += 1
        # batch_size = inputs.size(0)
        dice_follicle_meter.update(dice_follicle)
        dice_ovary_meter.update(dice_ovary)
    return dice_follicle_meter.avg, dice_ovary_meter.avg


if __name__ == '__main__':
    main()
