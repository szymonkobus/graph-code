from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from lossless import Paths


def get_prob(conf: Any, graph_size: int) -> Tensor:
    '''probability distribtion loading'''
    match conf.prob:
        case 'uniform':
            prob = torch.tensor([1/graph_size]).expand(graph_size)
        case 'linear':
            prob = polynomial_prob(graph_size, 1.)
        case 'quadratic':
            prob = polynomial_prob(graph_size, 2.)
        case 'sqrt':
            prob = polynomial_prob(graph_size, 0.5)
        case x:
            raise TypeError('prob type \'{}\' not implemented.'.format(x))
    if conf.prob_permute:
        prob = prob[torch.randperm(graph_size)]
    return prob

def polynomial_prob(size: int, exponent: float) -> Tensor:
    x = torch.arange(size, 0, -1)
    assert len(x)==size
    prob_unnormal = torch.pow(x, exponent)
    prob = prob_unnormal / torch.sum(prob_unnormal)
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

def first_paths_prob(paths: Paths, n_nodes: int, probs: Sequence[float]) \
        -> list[int]:
    '''assigns each vertex to first path it is a part of'''
    assign = [-1 for _ in range(n_nodes)]
    for i, path in enumerate(paths):
        for node in path:
            if assign[node]==-1:
                assign[node] = i
    assert not any([path==-1 for path in assign])
    return assign