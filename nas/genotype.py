"gonotype"
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    # 'sep_conv_5x5',
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
