from utils import count_parameters
from nas import RayNet_v0
m=RayNet_v0()
print(count_parameters(m))
