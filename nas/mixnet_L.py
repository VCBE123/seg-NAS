import timm
import torch
m=timm.create_model('mixnet_xl',pretrained=True,num_classes=3)
inputs=torch.randn([1,3,100,100])
out=m(inputs)
print(out.size())