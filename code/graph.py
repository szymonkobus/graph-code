from random import randint
from typing import Any

import torch
from scipy.sparse.csgraph import csgraph_from_dense
from torch import Tensor


class Graph:
    def __init__(self, adj: Tensor, node_name: list[str] | None = None,
                 adj_sparse = None):
        self.adj = adj
        self.node_name = node_name
        if node_name is None:
            self.node_name = [str(i) for i in range(len(self))]
        self.has_sparse = False
        self._sparse = adj_sparse

    @property
    def adj_sparse(self):
        if self._sparse is None:
            self._sparse = csgraph_from_dense(self.adj).astype(int)
            self.has_sparse = True
        return self._sparse

    def __len__(self):
        return self.adj.shape[0]

    def __str__(self):
        return self.adj.__str__()


def get_graph(conf: Any) -> Graph:
    '''graph loading'''
    match conf.graph_type:
        case 'grid':
            return create_grid(conf.dim)
        case 'random' | 'NLPA':
            return create_random_graph(conf.dim, conf.attachment_pow)
        case x:
            raise TypeError('Graph type \'{}\' not implemented.'.format(x))


def create_grid(dim: list[int]) -> Graph:
    base = [1]
    t = 1
    for i in dim:
        t *= i
        base.append(t)

    N = base[-1]
    adj = torch.zeros((N, N), dtype=torch.int)
    base_t = torch.tensor(base)
    for i in range(N):
        vec = int_to_vec(i, base_t)
        for k, _ in enumerate(dim):
            if vec[k] > 0:
                vec_t = vec.clone()
                vec_t[k] -= 1
                j = vec_to_int(vec_t, base_t)
                adj[i][j] = 1
                adj[j][i] = 1

    return Graph(adj)


def create_random_graph(dim: int, alpha: float, connect: bool = True) -> Graph:
    assert dim>1
    adj = torch.zeros((dim, dim), dtype=torch.int)
    adj[0,1] = 1
    adj[1,0] = 1
    neighbour_cnt = torch.zeros((dim,), dtype=torch.int)
    neighbour_cnt[:2] = 1
    for i in range(2, dim):
        weights = neighbour_cnt[:i].clone()
        if alpha != 1.0:
            weights = weights**alpha
        connect_prob = weights / torch.sum(weights)
        connection = torch.rand(i) < connect_prob
        neighbour_cnt[i] += torch.sum(connection)
        if connect and not torch.any(connection):
            connection[randint(0, i-1)] = 1
        neighbour_cnt[:i] += connection
        adj[i,:i] += connection
        adj[:i,i] += connection
    return Graph(adj)


def vec_to_int(vec: Tensor, base: Tensor) -> Tensor:
    return torch.sum(vec * base[:-1])


def int_to_vec(n: int, base: Tensor | list[int]) -> Tensor:
    vec = []
    for i in range(len(base)-1):
        d = n % base[i+1]
        vec.append(int(d / base[i]))
    return torch.tensor(vec)


def get_start(conf: Any, graph_size: int) -> int:
    '''returns index of starting vertex from config'''
    match conf.start:
        case 'uniform':
            return randint(0, graph_size)
        case int(i):
            return i
        case x:
            raise TypeError('Start type \'{}\' not implemented.'.format(x))
    