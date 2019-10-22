import sys
from RayNet import RayNet_v0
import torch
import os
import torch.nn as nn
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
f = open('log.txt', 'w')
sys.stdout = f


def get_hook_fn(i, device, phase):
    if 'pre_forward' in phase:
        def hook_fn(module, input):
            torch.cuda.empty_cache()
            print(f'{phase} {i} th-node:')
            max_mem = torch.cuda.max_memory_allocated(device=device)/1024/1024
            cur_mem = torch.cuda.memory_allocated(device=device)/1024/1024
            print(
                f'max_memory_allocated: {max_mem:.2f} memory_allocated: {cur_mem:.2f}',)
    else:
        def hook_fn(module, input, output):
            torch.cuda.empty_cache()
            print(f'{phase} {i} th-node:')
            max_mem = torch.cuda.max_memory_allocated(device=device)/1024/1024
            cur_mem = torch.cuda.memory_allocated(device=device)/1024/1024
            print(
                f'max_memory_allocated: {max_mem:.2f} memory_allocated: {cur_mem:.2f}')
    return hook_fn


def register(name, module):
    module.register_forward_hook(get_hook_fn(name, device, 'after_forward'))
    module.register_forward_pre_hook(get_hook_fn(name, device, 'pre_forward'))
    module.register_backward_hook(get_hook_fn(name, device, 'after_backward'))


def count_parameters(model):
    " coount parameters "
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


device = torch.device('cuda')
Criterion = nn.BCELoss().cuda()
net = RayNet_v0()
net = net.cuda()

# register("encode", net.encode)
# register("aspp", net.aspp)
# register("Raynet", net).
for i, model in net.named_children():
    register(i, model)

inputs = torch.randn(2, 3, 384, 384).cuda()
out = net(inputs)
target = torch.randn([2, 3, 384, 384]).cuda()
loss = Criterion(out, target)
loss.backward()

for i, model in net.named_children():
    print('{} param:{}'.format(i, count_parameters(model)))
f.close()
