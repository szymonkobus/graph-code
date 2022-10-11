from typing import Any, Callable

import torch
from torch import Tensor

from graph import Graph
from lossless import Paths
from lossy import first_paths_prob


def get_prob(conf: Any, graph: Graph) -> Tensor:
    '''probability distribtion loading'''
    match conf.prob:
        case 'uniform':
            prob = torch.tensor([1/len(graph)]).expand(len(graph))
        case x:
            raise TypeError('prob type \'{}\' not implemented.'.format(x))
    return prob


def get_path_prob(conf: Any) -> \
        Callable[[Paths, int, Sequence[float]], Sequence[int]]:
    '''returns probability distribtion of paths calculation method'''
    match conf.path_prob:
        case 'first':
            return first_paths_prob
        case x:
            raise TypeError('path_prob type \'{}\' not implemented.'.format(x))
     
     # additonal : 'random', 'iterate', 'optimal'