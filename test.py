
"train progress"
import os
import cv2
import time
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from nas import Unet
from dataloader import get_follicle
from utils import AverageMeter, get_dice
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='unet')
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='1')
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

    model = Unet(3, 3)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('model_best.pth.tar')['state_dice'])

    _, val_loader = get_follicle(ARGS)
    epoch_start = time.time()
    valid_dice = infer(val_loader, model)
    epoch_duration = time.time()-epoch_start
    print("valid_dice:{} duration: {}s".format(valid_dice, epoch_duration))


def infer(valid_loader, model):
    "validate func"
    dicemeter = AverageMeter()
    model.eval()
    count = 0
    for inputs, targets in tqdm.tqdm(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)

        dice = get_dice(logits, targets)
        pred = logits.cpu().numpy()
        om = np.argmax(pred.squeeze(), axis=0)
        om[om == 0] = 0
        om[om == 1] = 128
        om[om == 2] = 255
        print(om.shape)
        print(np.unique(om))

        cv2.imwrite('logs/{}.png'.format(count), om)
        count += 1
        batch_size = inputs.size(0)
        dicemeter.update(dice, batch_size)
    return dicemeter.avg


if __name__ == '__main__':
    main()
