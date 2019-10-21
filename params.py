from utils import count_parameters
import torch
from nas import RayNet
m=RayNet()

inputs= torch.randn(2,3,384,384)
out=m(inputs)
print(count_parameters(m))
print(out.size())
