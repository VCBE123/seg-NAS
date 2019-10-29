import matplotlib.pyplot as plt
from collections import OrderedDict
memo_txt = 'log.txt'
nodes = OrderedDict({'pre': {}, 'after': {}, 'memo_for': {}, 'params': {}})
txt = []
for line in open(memo_txt, 'r').readlines():
    txt.append(line)
print(len(txt))


for i in range(len(txt)):
    if 'pre_forward' in txt[i]:
        _, name, node = txt[i].split(' ')
        _, max_memo, _, memo = txt[i+1].split(' ')
        nodes['pre'][name] = {'max_memo': float(max_memo), 'memo': float(memo)}
    if 'after_forward' in txt[i]:
        _, name, node = txt[i].split(' ')
        _, max_memo, _, memo = txt[i+1].split(' ')
        nodes['after'][name] = {'max_memo': float(
            max_memo), 'memo': float(memo)}
        nodes['memo_for'][name] = nodes['after'][name]['max_memo']-nodes['pre'][name]['memo']

    if 'param:' in txt[i]:
        name = txt[i].split(' ')[0]
        nodes['params'][name] = float(txt[i].split(':')[1])

names = []
values = []
params = []
for k, v in nodes['memo_for'].items():
    names.append(k)
    values.append(nodes['memo_for'][k])
    params.append(nodes['params'][k])
# main figure

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(111)
l1 = ax1.bar(range(0, 2*len(names), 2), params,
             tick_label=None, color='purple')

ax1.set_ylabel('parameters (M)')
ticks = range(0, 2*len(names), 2)
ticks = [x+0.5 for x in ticks]
plt.xticks(ticks, names)
plt.xticks(rotation=90)
ax2 = ax1.twinx()

l2 = plt.bar(range(1, 2*len(names), 2), values,
             tick_label=None, color='darkcyan')
plt.xticks(rotation=90)
plt.ylabel('forward memory (MB)')
plt.legend(handles=[l1, l2], labels=['parameters', 'memory'])
plt.tight_layout()
plt.savefig('memory.png')
