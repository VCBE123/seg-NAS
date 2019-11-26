"train progress"
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
from nas import NASRayNetEval_v0, NASRayNetEval_v0_dense, WeightDiceLoss, ray2, ray3, NASRayNetEvalDense, MultipleOptimizer
from dataloader import get_follicle
from utils import AverageMeter, create_exp_dir, count_parameters, notice, save_checkpoint, get_dice_follicle, get_dice_ovary
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
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--save', type=str, default='logs')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='nasray_ray2_aspp_v0_dense')
    parser.add_argument('--lr_scheduler', default='step')
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='4,5,7')
    return parser.parse_args()


ARGS = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpus
ARGS.save = '{}/train-{}-{}-{}'.format(ARGS.save,
                                       ARGS.debug, ARGS.arch, time.strftime("%y%m%d-%h%m%s"))

create_exp_dir(ARGS.save, glob.glob('*.py')+glob.glob('*/*.py'))

LOG_FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=LOG_FORMAT, datefmt='%m/%d %h:%m:%s %p')

FH = logging.FileHandler(os.path.join(ARGS.save, 'log.txt'))
FH.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(FH)
WRITER = SummaryWriter(log_dir=os.path.dirname(
    logging.Logger.root.handlers[1].baseFilename))


def main():
    "train main"
    np.random.seed(ARGS.seed)
    cudnn.benchmark = True
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)

    logging.info("args=%s", ARGS)
    num_gpus = torch.cuda.device_count()
    logging.info("using gpus: %d", num_gpus)
    model = NASRayNetEval_v0_dense(genotype=ray2)
    model = nn.DataParallel(model)
    model = model.cuda()

    logging.info("params size = %f m", count_parameters(model))

    encoder_parameters = []
    decoder_parameters = []
    for k, parameter in model.named_parameters():
        if 'encode' in k:
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)
    optimizer_encoder = torch.optim.SGD(
        encoder_parameters, 0.003, momentum=ARGS.momentum, weight_decay=1e-4)
    optimizer_decoder = torch.optim.SGD(
        decoder_parameters, 0.003, weight_decay=1e-4, momentum=ARGS.momentum)
    multop = MultipleOptimizer(optimizer_decoder, optimizer_encoder)

    lr_scheduler_encoder = torch.optim.lr_scheduler.StepLR(
        optimizer_encoder, step_size=50, gamma=0.1)
    lr_scheduler_decoder = torch.optim.lr_scheduler.StepLR(
        optimizer_decoder, step_size=50, gamma=0.1)
    criterion = WeightDiceLoss().cuda()

    multop = MultipleOptimizer(optimizer_decoder, optimizer_encoder)
    criterion = WeightDiceLoss().cuda()
    train_loader, val_loader = get_follicle(16, 8, train_aug=True)
    best_dice = 0
    for epoch in range(ARGS.epochs):
        current_lr_encoder = optimizer_encoder.param_groups[0]['lr']
        current_lr_decoder = optimizer_decoder.param_groups[0]['lr']

        logging.info("epoch: %d lr_for encoder %e lr_for decoder %e",
                     epoch, current_lr_encoder, current_lr_decoder)
        epoch_start = time.time()
        train_loss = train(
            train_loader, model, criterion, multop)

        lr_scheduler_decoder.step()
        lr_scheduler_encoder.step()

        WRITER.add_scalars('loss', {'train_loss': train_loss}, epoch)
        logging.info("train_loss: %f", train_loss)

        valid_dice_follicle, valid_dice_ovary, valid_loss = infer(
            val_loader, model, criterion)
        logging.info("valid_dice_follicle: %f valid_dice_ovary: %f",
                     valid_dice_follicle, valid_dice_ovary)
        logging.info("valid_loss: %f", valid_loss)

        WRITER.add_scalars(
            'dice', {'valid_dice_ovary': valid_dice_ovary}, epoch)
        WRITER.add_scalars(
            'dice', {'valid_dice_follicle': valid_dice_follicle}, epoch)
        WRITER.add_scalars('loss', {'valid_loss': valid_loss}, epoch)

        epoch_duration = time.time()-epoch_start
        logging.info("epoch time: %ds.", epoch_duration)

        is_best = False
        if valid_dice_ovary > best_dice:
            best_dice = valid_dice_ovary
            is_best = True
            try:
                notice('validation-unet',
                       message="epoch:{} best_dice:{}".format(epoch, best_dice))
            finally:
                pass
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(),
                         'best_dice': best_dice, 'optimizer': optimizer_encoder.state_dict()}, is_best, ARGS.save)
    logging.info("Best finaly ovary dice: %e", best_dice)


def train(train_loader, model, criterion, optimizer):
    "training func"
    objs = AverageMeter()
    # dicemeter = AverageMeter()
    batch_time = AverageMeter()

    model.train()
    optimizer.zero_grad()
    accumulate = 6
    for step, (inputs, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        b_start = time.time()
        logits = model(inputs)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.module.parameters(), ARGS.grad_clip)
        if step % accumulate:
            optimizer.step()
        batch_size = inputs.size(0)
        batch_time.update(time.time()-b_start)
        objs.update(loss, batch_size)
        if step % ARGS.report == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time-start_time
                start_time = time.time()
            logging.info('Train Step: %03d Loss: %e  Duration: %ds BTime: %.3fs',
                         step, objs.avg, duration, batch_time.avg)

    return objs.avg


def infer(valid_loader, model, criterion):
    "validate func"
    objs = AverageMeter()
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    model.eval()
    for step, (inputs, targets) in enumerate(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, targets)
        dice_follicle = get_dice_follicle(logits, targets)
        dice_ovary = get_dice_ovary(logits, targets)
        batch_size = inputs.size(0)

        objs.update(loss, batch_size)
        dice_follicle_meter.update(dice_follicle, batch_size)
        dice_ovary_meter.update(dice_ovary, batch_size)
        if step % ARGS.report == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time-start_time
                start_time = time.time()
            logging.info("Valid Step: %03d Objs: %e Follicle_Dice: %e Overay_Dice: %e Duration: %ds",
                         step, objs.avg, dice_follicle_meter.avg, dice_ovary_meter.avg, duration)
    return dice_follicle_meter.avg, dice_ovary_meter.avg, objs.avg,


if __name__ == '__main__':
    main()