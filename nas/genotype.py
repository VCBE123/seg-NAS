"gonotype"
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    # 'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'se',
    'md_conv_35',
    'md_conv_357'
]
s3 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('sep_conv_5x5', 4)], normal_concat=range(
    2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

u1 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('se', 0), ('md_conv_35', 1), ('dil_conv_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6),
              reduce=[('sep_conv_3x3', 0), ('md_conv_357', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('se', 3), ('se', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
ray1 = Genotype(normal=[('se', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('se', 0), ('dil_conv_5x5', 3), ('se', 3), ('md_conv_357', 4)], normal_concat=range(2, 6),
                reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('md_conv_35', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
ray2 = Genotype(normal=[('se', 0), ('sep_conv_3x3', 1), ('se', 0), ('se', 1), ('se', 0), ('md_conv_357', 3), ('se', 0), ('md_conv_35', 2)], normal_concat=range(2, 6), reduce=[
                ('sep_conv_5x5', 0), ('md_conv_357', 1), ('dil_conv_3x3', 1), ('md_conv_357', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
ray3= Genotype(normal=[('se', 0), ('se', 1), ('sep_conv_3x3', 1), ('md_conv_357', 2), ('se', 0), ('se', 1), ('se', 0), ('md_conv_35', 3)], normal_concat=range(2, 6), reduce=[('se', 0),
 ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('md_conv_357', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

sk1 =Genotype(normal=[('se', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 3), ('skip_connect', 1), ('se', 4)], normal_concat=range(2, 6),
 reduce=[('sep_conv_5x5', 0), ('md_conv_357', 1), ('se', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('md_conv_357', 2), ('se', 3)], reduce_concat=range(2, 6))