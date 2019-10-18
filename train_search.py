"train search"
import copy
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
from nas import NASUnet, WeightDiceLoss, PRIMITIVES
from dataloader import get_follicle, FollicleDataset, ImgAugTrans
from utils import AverageMeter, create_exp_dir, count_parameters, notice, save_checkpoint, get_dice_follicle, get_dice_ovary
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_min', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--report', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save', type=str, default='logs')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--train_portion', default=0.7,
                        help='the partion for update weights')
    parser.add_argument('--arch', default='nasunet')
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4)
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3)

    parser.add_argument('--inital_channel', type=int, default=8)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--lr_scheduler', default='step')
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='0,1,2,3,5')
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

    train_trans = ImgAugTrans(384)
    traindata = FollicleDataset('/data/follicle/train.txt', train_trans)
    num_train = len(traindata)
    indices = list(range(num_train))
    split = int(np.floor(ARGS.train_portion*num_train))

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=ARGS.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(
        indices[:split]), pin_memory=True, num_workers=ARGS.workers)
    valid_loader = torch.utils.data.DataLoader(traindata, batch_size=ARGS.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(
        indices[split:]), pin_memory=True, num_workers=ARGS.workers)

    criterion = WeightDiceLoss().cuda()
    switches = []
    for _ in range(14):
        switches.append([True for j, _ in enumerate(PRIMITIVES)])

    switches_norm = copy.deepcopy(switches)
    switches_redu = copy.deepcopy(switches)

    model = NASUnet(ARGS.inital_channel, ARGS.classes, ARGS.layers, criterion,
                    4, switches_normal=switches_norm, switches_reduce=switches_redu)
    model = nn.DataParallel(model)
    model = model.cuda()

    logging.info("params size = %f m", count_parameters(model))
    network_params = []
    for k, v in model.named_parameters():
        if not(k.endswith('alpha_normal') or k.endswith('alpha_reduce')):
            network_params.append(v)

    optimizer = torch.optim.SGD(model.parameters(
    ), ARGS.learning_rate, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay)

    optimizer_arch = torch.optim.Adam(model.module.arch_parameters(
    ), lr=ARGS.arch_learning_rate, betas=(0.5, 0.999), weight_decay=ARGS.arch_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, ARGS.epochs, eta_min=ARGS.learning_rate_min)

    # criterion = torch.nn.BCELoss().cuda()
    criterion = WeightDiceLoss().cuda()
    train_loader, val_loader = get_follicle(ARGS)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    best_dice = 0

    for epoch in range(ARGS.epochs):
        current_lr = scheduler.get_lr()[0]
        logging.info("epoch: %d lr %e", epoch, current_lr)
        epoch_start = time.time()
        if epoch<1:
            train_arch=False
        else:
            train_arch=True
                
        train_loss = train(
            train_loader,valid_loader, model,network_params, criterion, optimizer,optimizer_arch,current_lr,train_arch=train_arch)

        WRITER.add_scalars('loss', {'train_loss': train_loss}, epoch)
        logging.info("train_loss: %f", train_loss)
        if ARGS.epochs-epoch<5:
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
            if valid_dice_ovary > best_dice:
                best_dice = valid_dice_ovary
                is_best = True
                try:
                    notice('validation-nasunet',
                        message="epoch:{} best_dice:{}".format(epoch, best_dice))
                finally:
                    pass


        epoch_duration = time.time()-epoch_start
        logging.info("epoch time: %ds.", epoch_duration)
        scheduler.step()
    


    torch.save(model,os.path.join(ARGS.save, 'weights.pt'))
    arch_param = model.module.arch_parameters()
    normal_prob=F.softmax(arch_param[0], dim=-1).data.cpu().numpy()
    for i in range(14):
        idxs=[]
        for j,_ in enumerate(PRIMITIVES):
            if switches_norm[i][j]:
                idxs.append(j)
    
    reduce_prob= F.softmax(arch_param[1],dim=-1).data.cpu().numpy()
    for j in range(14):
        idxs=[]
        for j,_ in enumerate(PRIMITIVES):
            if switches_redu[i][j]:
                idxs.append(j)
    logging.info('switches_normal = %s', switches_norm)  # log the status of the switches
    logging_switches(switches_norm)
    logging.info('switches_reduce = %s', switches_redu)
    logging_switches(switches_redu)
    genotype = parse_network(switches_norm, switches_redu)
    logging.info(genotype)   # this is the output genotype
            ## restrict skipconnect (normal cell only)
    logging.info(' Search finish...')
     


def train(train_loader, valid_loader, model, network_params, criterion, optimizer, optimizer_arch, lr, train_arch=True):
    "training func"
    objs = AverageMeter()
    # dicemeter = AverageMeter()
    batch_time = AverageMeter()

    model.train()
    optimizer.zero_grad()
    for step, (inputs, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        b_start = time.time()
        if train_arch:
            try:
                inputs_search, target_search = next(valid_loader_iter)
            except:
                valid_loader_iter = iter(valid_loader)
                inputs_search, target_search = next(valid_loader_iter)

            inputs_search, target_search = inputs_search.cuda(
            ), target_search.cuda(non_blocking=True)
            optimizer_arch.zero_grad()
            logits = model(inputs_search)
            loss_arch = criterion(logits, target_search)
            loss_arch.backward()
            nn.utils.clip_grad_norm_(
                model.modules.arch_parameters(), ARGS.grad_clip)
            optimizer_arch.step()

        logits = model(inputs)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network_params, ARGS.grad_clip)
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


def parse_network(switches_normal, switches_reduce):
    # finish parse
    def _parse_switches(switches):
        n = 2                                     # 2,3,4,5 = 14
        start = 0
        gene = []
        step = 4
        for _ in range(step):  # 0,1,2,3
            end = start + n
            # [0,1] [2,3,4],[5,6,7,8],[9,10,11,12,13]
            for j in range(start, end):
                for k in range(len(switches[j])):                   # [11]
                    if switches[j][k]:
                        # [k], [0,1] [0,1,2],[0,1,2,3],[0,1,2,3,4]
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for _ in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for _ in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


if __name__ == '__main__':
    main()
