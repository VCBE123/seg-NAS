"search code for the original p-search"
import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import copy
from nas import WeightDiceLoss, PRIMITIVES, Genotype, NASRayNet
from dataloader import FollicleDataset, ImgAugTrans
from utils import AverageMeter, create_exp_dir, count_parameters, notice, get_dice_follicle, get_dice_ovary

parser = argparse.ArgumentParser("p-search nas- mutant")
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.000025, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpus', type=str,
                    default='0,1,2,3', help='GPU device id')
parser.add_argument('--epochs', type=int, default=25,
                    help='num of training epochs')
parser.add_argument('--mutant', default=[5,5,5], help='mutant point')
parser.add_argument('--layers', type=int, default=12,
                    help='total number of layers')
parser.add_argument('--save', type=str, default='logs', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.7, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float,
                    default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--note', type=str, default='nas-mutation',
                    help='note for this run')
parser.add_argument('--debug', default='')
parser.add_argument('--arch', default='nasray-mutate')
parser.add_argument('--classes', default=3)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.save = '{}/train-{}-{}-{}'.format(args.save,
                                       args.debug, args.arch, time.strftime("%y%m%d-%h%m%s"))

create_exp_dir(args.save, glob.glob('*.py')+glob.glob('*/*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(log_dir=os.path.dirname(
    logging.Logger.root.handlers[1].baseFilename))


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('GPU number = %d' % torch.cuda.device_count())
    logging.info("args = %s", args)
    #  prepare dataset

    train_trans = ImgAugTrans(384)
    train_data = FollicleDataset('/data/lir/follicle/train_pain.txt', train_trans)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers,drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:]),
        pin_memory=True, num_workers=args.workers,drop_last=True)

    # build Network
    criterion = WeightDiceLoss().cuda()

    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])  # [14*8]
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    eps_no_archs = [3, 3, 3, 3]

    for sp in range(len(num_to_keep)):

        # model = NASUnet(args.init_channels, args.classes, args.layers, criterion,
                        # 4, switches_normal=switches_normal, switches_reduce=switches_reduce)
        model = NASRayNet(switches_normal=switches_normal,
                          switches_expansion=switches_reduce)

        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", count_parameters(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)
        optimizer = torch.optim.SGD(
            network_params,
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                       lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        best_dice = 0
        for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                train_obj = train(train_queue, valid_queue, model, network_params,
                                  criterion, optimizer, optimizer_a, lr, train_arch=False)
            else:
                train_obj = train(train_queue, valid_queue, model, network_params,
                                  criterion, optimizer, optimizer_a, lr, train_arch=True)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_dice_follicle, valid_dice_ovary, valid_loss = infer(
                    valid_queue, model, criterion)
                logging.info("valid_dice_follicle: %f valid_dice_ovary: %f",
                             valid_dice_follicle, valid_dice_ovary)
                logging.info("valid_loss: %f", valid_loss)

                writer.add_scalars(
                    'dice', {'valid_dice_ovary': valid_dice_ovary}, epoch)
                writer.add_scalars(
                    'dice', {'valid_dice_follicle': valid_dice_follicle}, epoch)
                writer.add_scalars('loss', {'valid_loss': valid_loss}, epoch)
                if valid_dice_ovary > best_dice:
                    best_dice = valid_dice_ovary
                    is_best = True
                    try:
                        notice('validation-nasunet',
                               message="epoch:{} best_dice:{}".format(epoch, best_dice))
                    finally:
                        pass
        torch.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(
                    normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(
                    reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(
                arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(
                arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            # restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(
                        switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(
                        switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)

    try:
        notice('finish search nasunet', message="epoch:{}".format(epoch))
    finally:
        pass


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = AverageMeter()
    batch_time = AverageMeter()

    model.train()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        b_start = time.time()
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(
                model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        objs.update(loss.data.item(), n)
        batch_time.update(time.time()-b_start)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time-start_time
                start_time = time.time()

            logging.info('TRAIN Step: %03d loss: %e Duration :%ds BTime %ds',
                         step, objs.avg, duration, batch_time.avg)

    return objs.avg


def infer(valid_queue, model, criterion):
    objs = AverageMeter()
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        dice_follicle = get_dice_follicle(logits, target)
        dice_ovary = get_dice_ovary(logits, target)

        n = input.size(0)
        objs.update(loss.data.item(), n)
        dice_follicle_meter.update(dice_follicle, n)
        dice_ovary_meter.update(dice_ovary, n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time-start_time
                start_time = time.time

            logging.info('valid Step %03d Loss: %e Follicle_Dice: %e Overay_Dice: %e Duration: %ds ',
                         step, objs.avg, dice_follicle_meter.avg, dice_ovary_meter.avg, duration)

    return dice_follicle_meter.avg, dice_ovary_meter.avg, objs.avg


def parse_network(switches_normal, switches_reduce):
    # todo understand below func
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for _ in range(step):
            end = start + n
            for j in range(start, end):  # step1 0:2 step2 2:4 step3 4:6 step4 6:8  (01234567)
                for k in range(len(switches[j])):   # 8
                    if switches[j][k]:              # 0:[0:14]
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
    """
    """
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


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches


if __name__ == '__main__':

    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
