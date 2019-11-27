
"train progress with Ray"
import logging
import os
import time
import argparse
import glob
import sys

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from nas import NASRayNetEval_v0, NASRayNetEval_v0_dense, WeightDiceLoss, ray2, ray3, MultipleOptimizer
from dataloader import get_follicle
from utils import AverageMeter, create_exp_dir, count_parameters, notice, save_checkpoint, get_dice_follicle, get_dice_ovary

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--report', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='nasray_ray2_aspp_4cell')
    parser.add_argument('--lr_scheduler', default='step')
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='0,1')
    return parser.parse_args()


ARGS = get_parser()
np.random.seed(ARGS.seed)
cudnn.benchmark = True
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed(ARGS.seed)


def train_follicle_dense(config):
    "train follicle"
    if config["Dense"]:
        model = NASRayNetEval_v0_dense(genotype=ray2)
    else:
        model = NASRayNetEval_v0(genotype=ray2)
    model = nn.DataParallel(model)
    model = model.cuda()
    encoder_parameters = []
    decoder_parameters = []
    for k, parameter in model.named_parameters():
        if 'encode' in k:
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)

    optimizer_encoder = torch.optim.Adam( encoder_parameters, config["learning_rate"], weight_decay=config["weight_decay"]) 
    optimizer_decoder = torch.optim.Adam( decoder_parameters, config["learning_rate"]*config["times"], weight_decay=config["weight_decay"])
    multop = MultipleOptimizer(optimizer_decoder, optimizer_encoder)
    criterion = WeightDiceLoss().cuda()
    train_loader, val_loader = get_follicle(
        config["batch_size"], 8, train_aug=True)
    best_dice_follicle = 0
    for epoch in range(ARGS.epochs):
        train_loss = train(train_loader, model, criterion,
                           multop, accumulate=config["accumulate"])
        valid_dice_follicle, valid_dice_ovary, valid_loss = infer(
            val_loader, model, criterion)
        is_best = False
        if valid_dice_follicle > best_dice_follicle:
            best_dice_follicle = valid_dice_follicle
            is_best = True
            if valid_dice_follicle > 0.89:
                try:
                    notice('validation-search',
                           message="epoch:{} best_dice:{}".format(epoch, best_dice_follicle))
                finally:
                    pass
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(),
                         'best_dice_follicle': best_dice_follicle, 'optimizer': optimizer_decoder.state_dict()}, is_best, "")
        track.log( train_loss=train_loss, valid_loss=valid_loss, valid_dice_follicle=valid_dice_follicle, valid_dice_ovary=valid_dice_ovary, best_dice_follicle=best_dice_follicle)


def train(train_loader, model, criterion, optimizer, accumulate):
    "training func"
    objs = AverageMeter()
    model.train()
    optimizer.zero_grad()
    for step, (inputs, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        if step % accumulate:
            optimizer.step()
        objs.update(loss.data.item(), inputs.size(0))
    return objs.avg


def infer(valid_loader, model, criterion):
    "validate func"
    objs = AverageMeter()
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    model.eval()
    for _, (inputs, targets) in enumerate(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, targets)
        dice_follicle = get_dice_follicle(logits, targets)
        dice_ovary = get_dice_ovary(logits, targets)
        batch_size = inputs.size(0)
        objs.update(loss.data.item(), batch_size)
        dice_follicle_meter.update(dice_follicle, batch_size)
        dice_ovary_meter.update(dice_ovary, batch_size)
    return dice_follicle_meter.avg, dice_ovary_meter.avg, objs.avg,


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6'
    ray.init(num_gpus=6, ignore_reinit_error=True)
    # sched = ASHAScheduler(metric="best_dice_follicle", mode="max")
    search_space = {
        "Dense":tune.choice([True,False]),
        "learning_rate": tune.choice([ 1e-4 ]),
                    "weight_decay": tune.choice([ 1e-5]),
                    "accumulate": tune.choice([  6]),
                    "batch_size": tune.choice([16]),
                    "times": tune.choice([ 2])}
    analysis = tune.run( train_follicle_dense, num_samples=100,
        stop={"best_dice_follicle": 90.0},
        resources_per_trial={"cpu": 8, "gpu": 3}, config=search_space)
    print("Best config:", analysis.get_best_config(metric="best_dice_follicle"))
