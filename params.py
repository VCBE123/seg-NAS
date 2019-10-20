import timm
from utils import count_parameters

m=timm.create_model('mixnet_xl',pretrained=True,num_classes=3)
print(count_parameters(m))
