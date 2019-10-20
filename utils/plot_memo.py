import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
memo_txt = 'log.txt'
nodes = OrderedDict({'pre': {}, 'after': {}, 'memo_for': {}, 'max_memo': {
}, 'back': {}, 'memo_back': {}, 'memo_for_new': {}})
forward = []
backward = []
max = []
txt = []
f_memo = []
f_max = []
for line in open(memo_txt, 'r').readlines():
    txt.append(line)
print(len(txt))
params = []
for i in range(len(txt)):
    if 'pre_forward' in txt[i]:
        _, name, node = txt[i].split(' ')
        _, max_memo, _, memo = txt[i+1].split(' ')
        nodes['pre'][name +
                     node] = {'max_memo': float(max_memo), 'memo': float(memo)}
        # max.append(max_memo)
    if 'after_forward' in txt[i]:
        _, name, node = txt[i].split(' ')
        _, max_memo, _, memo = txt[i+1].split(' ')
        nodes['after'][name +
                       node] = {'max_memo': float(max_memo), 'memo': float(memo)}
        nodes['memo_for'][name+node] = nodes['after'][name +
                                                      node]['memo']-nodes['pre'][name+node]['memo']
        nodes['memo_for_new'][name+node] = nodes['after'][name +
                                                          node]['max_memo']-nodes['pre'][name+node]['memo']

        nodes['max_memo'][name+node] = nodes['after'][name+node]['max_memo']
        forward.append(nodes['after'][name+node]
                       ['max_memo']-nodes['pre'][name+node]['memo'])
        # forward_new.append(nodes['after'][name+node]['max_memo']-nodes['pre'][name+node]['memo'])
        f_max.append(nodes['after'][name+node]['max_memo'])
        f_memo.append(nodes['after'][name+node]['memo'])
        max.append(max_memo)
    if 'after_backward' in txt[i]:
        _, name, node = txt[i].split(' ')
        _, max_memo, _, memo = txt[i + 1].split(' ')
        nodes['back'][name +
                      node] = {'max_memo': float(max_memo), 'memo': float(memo)}
        nodes['memo_back'][name + node] = nodes['after'][name +
                                                         node]['memo'] - nodes['pre'][name + node]['memo']
        nodes['max_memo'][name + node] = nodes['after'][name + node]['max_memo']
        backward.append(nodes['back'][name+node]['memo'] -
                        nodes['after'][name+node]['memo'])
        max.append(max_memo)
        # after_max.append(float(max_memo))
    if 'param:' in txt[i]:
        params.append(float(txt[i].split(':')[1]))
# forward DarkUnit
names = []
values = []
values_new = []
for k, v in nodes['memo_for'].items():
    names.append(k)
    values.append(v)
    values_new.append(nodes['memo_for_new'][k])
# main figure

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(111)
l1 = ax1.bar(range(0, 2*len(names), 2), params, tick_label=None, color='purple')
# l1=ax1.plot(range(0,len(names)),params,color='g')
ax1.set_ylabel('parameters (M)')
ticks = range(0, 2*len(names), 2)
ticks = [x+0.5 for x in ticks]
plt.xticks(ticks, names)
plt.xticks(rotation=90)
ax2 = ax1.twinx()
# l2=ax2.plot(range(0,len(names)),values_new,color='b')

l2 = plt.bar(range(1, 2*len(names), 2), values_new,
             tick_label=None, color='darkcyan')
plt.xticks(rotation=90)
# plt.xlabel('forward memory')
plt.ylabel('forward memory (MB)')
plt.legend(handles=[l1, l2], labels=['parameters', 'memory'])
plt.tight_layout()
plt.savefig('memory.png')
# plt.show()
