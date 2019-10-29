"utils"
import os
import shutil
import torch
import requests
import numpy as np


def create_exp_dir(path, scripts_to_save=None):
    "create logdir and save .py"
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(
                path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def notice(title='', message=''):
    "send message to my wechat"
    url = 'https://sc.ftqq.com/SCU59598T67062eca56424fa420ece6b0fa3ed62c5d74a64e7de40.send?'
    params = {'text': title, "desp": message}
    requests.post(url=url, params=params)


def count_parameters(model):
    " coount parameters "
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def save_checkpoint(state, is_best, save):
    "save checkpoint"
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
