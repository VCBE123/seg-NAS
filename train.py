import argparse
import numpy as np
import torch
from logging import getLogger
import shutil
from tensorboardX import SummaryWriter
import os
import time
import yaml
from dataset import get_composed_augmentations
import torchvision
import numpy
import torch.nn as nn
from nas import dice_loss,Unet,init_weithts
from tqdm import tqdm
from dataset import get_loader
import torch.optim  as opt
from torch.optim import lr_scheduler
from utils import averageMeter,bestMeter
from utils import get_dice

def train(cfg,writer,logger):
	torch.manual_seed(cfg.get("seed",42))
	torch.cuda.manual_seed(cfg.get("seed",42))
	np.random.seed(cfg.get("seed",42))

	augmentation=get_composed_augmentations(cfg['training'].get("augmentation",None))
	batch_size=cfg.get('batch_size',16)
	train_loader=get_loader(cfg["data"]['train'],batch_size,augment=augmentation,shuffle=True)
	val_loader=get_loader(cfg['data']['valid'],batch_size)

	model=Unet()
	# init_weithts(model)
	model=model.cuda()
	model=nn.DataParallel(model)

	optimizer=opt.SGD(model.parameters(),lr=cfg.get('lr',1e-3),momentum=0.9,weight_decay=cfg.get('weight',1e-4))
	scheduler=lr_scheduler.StepLR(optimizer,step_size=cfg.get('step',30))
	logger.info(cfg)
	start_epoch=1
	if cfg['training']["resume"]:
		if os.path.isfile(cfg['training']['resume']):
			logger.info("load model from checkpoint {}".format(cfg['training']['resume']))
			checkpoint=torch.load(cfg['training']['resume'])
			model.load_state_dict(checkpoint['model_state'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.load_state_dict(checkpoint['scheduler'])
			start_epoch=checkpoint['epoch']
			logger.info("checkpoint epoch: {}, dice :{.2f}".format(checkpoint['epoch'],checkpoint['dice']))
	else:
		logger.info('train from scratch')
	train_loss_meter=averageMeter()
	val_loss_meter=averageMeter()
	val_dice_meter=averageMeter()
	best_dice=bestMeter()
	for i in range(start_epoch,cfg.get('epoch',60)+1):
		model.train()
		for image,mask in train_loader:
			scheduler.step(i)
			logger.info("lr :{}".format(scheduler.get_lr()))
			image,mask=image.cuda(),mask.cuda()
			optimizer.zero_grad()
			outputs=model(image)
			loss=dice_loss(outputs,mask)
			train_loss_meter.update(loss.item())
			loss.backward()
			optimizer.step()
		fmt_str="epoch [{:d}/{:d}] train loss : {:.4f}"
		print_str=fmt_str.format(i,cfg['epoch'],train_loss_meter.avg)
		logger.info(print_str)
		print(print_str)
		writer.add_scalar("train_loss",train_loss_meter.avg,i)
		train_loss_meter.reset()

		if i% cfg['training']["val_interval"]==0 or i==cfg['training']['epoch']:
			model.eval()
			for image,mask in ( val_loader):
				image,mask=image.cuda(),mask.cuda()
				outputs=model(image)
				val_loss=dice_loss(outputs,mask)
				val_loss_meter.update(val_loss.item())
				val_dice_meter.update(get_dice(outputs.cpu().detach().numpy(),mask.cpu().detach().numpy()))
			if best_dice.update(val_dice_meter.avg):
				logger.info('best dice update to {}'.format(best_dice.best))
				print('best dice update to {}'.format(best_dice.best))
				state={
					'epoch':i,
					'model_state':model.state_dict(),
					'optimizer_state':optimizer.state_dict(),
					'scheduler_state':scheduler.state_dict(),
					'best_dice': best_dice.best
				}
				torch.save(state,os.path.join(writer.file_writer.get_logdir(),"best_epoch.checkpoint"))
			writer.add_scalar('val_loss',val_loss_meter.avg,i)
			writer.add_scalar('val_dice',val_dice_meter.avg)
			logger.info('epoch {} Val loss : {:.4f}'.format(i,val_loss_meter.avg))
			print('epoch {} Val loss : {:.4f}'.format(i,val_loss_meter.avg))
			val_loss_meter.reset()



if __name__ == '__main__':
	parser=argparse.ArgumentParser(description='Train segmentation')
	parser.add_argument('--config',nargs='?',type=str,default='config/seg.yaml',
						help='configure path')
	args=parser.parse_args()

	with open(args.config) as fp:
		cfg=yaml.load(fp,Loader=yaml.FullLoader)
	os.environ["CUDA_VISIBLE_DEVICES"]=cfg.get("gpu_id",'2,3')
	ti=time.time().__str__()
	logdir=os.path.join('logdir',ti)
	writer=SummaryWriter(log_dir=logdir)

	shutil.copy(args.config,logdir)
	logger=getLogger(logdir)
	train(cfg,writer,logger)







