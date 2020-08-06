
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
from nas import NASRayNetEval_aspp_base, ray2, RayNet_v0, ray3, RayNet_v1,NASRayNetEval_aspp_2,Unet

from dataloader import get_follicle
from utils import AverageMeter, get_dice_ovary, get_dice_follicle, get_hd,hd95,asd,mean_IU
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def get_parser():
    "parser argument"
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--arch', default='low')
    parser.add_argument('--classes', default=3)
    parser.add_argument('--debug', default='')
    parser.add_argument('--gpus', default='4,5')
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

    # model =NASRayNetEval_aspp_2(genotype=ray2)
    model = NASRayNetEval_aspp_base(genotype=ray2)
    model=Unet(3,3)
    model = nn.DataParallel(model)
    model = model.cuda()

    state_dict = torch.load('logs/train--unet-200804-Aug081596554193/checkpoint.pth.tar')
    # state_dict = torch.load('exp2/train--nasray_ray2_ASPP_Cell_low_stack-191207-Dec121575729910/model_best.pth.tar')
    model.load_state_dict(state_dict['state_dict'])
    model = model.module

    _, val_loader = get_follicle(1,1,return_path=True)
    epoch_start = time.time()
    valid_dice_follicle, valid_dice_ovary, valid_hd_ovary,valid_hd_follicle, valid_jc_ovary, valid_jc_follicle,valid_asd_ovary, valid_asd_follicle = infer(val_loader, model)
    epoch_duration = time.time()-epoch_start
    print(" valid_dice_Ovary:{} vald_dice_follicle:{} \n valid_hd_ovary:{} valid_hd_follicle:{} \n valid_jc_ovary:{} valid_jc_follicle:{} \n valid_asd_ovary:{} valid_asd_follicle:{} duration: {}s".format(
        valid_dice_ovary, valid_dice_follicle, valid_hd_ovary,valid_hd_follicle, valid_jc_ovary, valid_jc_follicle, valid_asd_ovary, valid_asd_follicle, epoch_duration))


def infer(valid_loader, model):
    "validate func"
    dice_ovary_meter = AverageMeter()
    dice_follicle_meter = AverageMeter()
    hd_meter_ovary = AverageMeter()
    hd_meter_follicle = AverageMeter()
    asd_meter_ovary = AverageMeter()
    asd_meter_follicle = AverageMeter()

    jc_meter_ovary=AverageMeter()
    jc_meter_follicle=AverageMeter()
    # for m in model.modules():
    # if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    # m.track_running_stats = False

    model.eval()
    count = 0
    for inputs, targets,image_path,label_path in tqdm.tqdm(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
        dice_follicle,jc_follicle = get_dice_follicle(logits, targets,eval=True)
        dice_ovary, jc_ovary = get_dice_ovary(logits, targets,eval=True)

        # hd = get_hd(logits, targets)
        hd_ovary,hd_follicle, asd_ovary,asd_follicle=hd95(logits,targets)

        save_mask= False
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
        hd_meter_ovary.update(hd_ovary, batch_size)
        if not hd_follicle == np.inf:
            hd_meter_follicle.update(hd_follicle, batch_size)

        if not asd_follicle == np.inf:
            asd_meter_follicle.update(asd_follicle, batch_size)
        asd_meter_ovary.update(asd_ovary, batch_size)

        jc_meter_ovary.update(jc_ovary,batch_size)
        jc_meter_follicle.update(jc_follicle,batch_size)
    return dice_follicle_meter.avg, dice_ovary_meter.avg, hd_meter_ovary.avg,hd_meter_follicle.avg,jc_meter_ovary.avg,jc_meter_follicle.avg, asd_meter_ovary.avg,asd_meter_follicle.avg


if __name__ == '__main__':
    main()
