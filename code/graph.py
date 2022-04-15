import torch
from torch import Tensor


class Graph:
    def __init__(self, adj: Tensor, node_name: list[str]|None = None):
        self.adj = adj
        self.node_name = node_name
        if node_name is None:
            self.node_name = [str(i) for i in range(len(self))]

    def __len__(self):
        return self.adj.shape[0]

    def __str__(self):
        return self.adj.__str__()

# graph loading

def get_graph(conf) -> Graph:
    match conf.graph_type:
        case 'grid':
            return create_grid(conf.dim)
        case x:
            raise TypeError('Graph type \'{}\' not implemented.'.format(x))

def create_grid(dim: list[int]) -> Graph:
    base = [1]
    t = 1
    for i in dim:
        t *= i
        base.append(t)
    
    N = base[-1]
    adj = torch.zeros((N, N), dtype=int)
    base_t = torch.tensor(base)
    for i in range(N):
        vec = int_to_vec(i, base_t)
        for k, d in enumerate(dim):
            if vec[k] > 0:
                vec_t = vec.clone()
                vec_t[k] -= 1
                j = vec_to_int(vec_t, base_t)
                adj[i][j] = 1
                adj[j][i] = 1

    return Graph(adj)

def vec_to_int(vec: Tensor, base: Tensor) -> Tensor:
    return torch.sum(vec * base[:-1])

def int_to_vec(n: int, base: Tensor | list[int]) -> Tensor:
    vec = []
    for i in range(len(base)-1):
        d = n % base[i+1]
        vec.append(int(d / base[i]))
    return torch.tensor(vec)
