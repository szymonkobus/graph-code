import numpy as np
import scipy.sparse.csgraph
import torch
from torch import Tensor

from comm import Comm, comm_dist_iterator
from graph import Graph


def distance_bound(graph: Graph, start: int, prob: Tensor) -> float:
    distances = node_distance(graph, start)
    return torch.sum(prob * distances).item()


def node_distance(graph: Graph, start: int) -> Tensor:
    distances = scipy.sparse.csgraph.shortest_path(graph.adj_sparse, 
                                                   indices=start, method='D')
    distances[distances==np.inf] = -1
    return torch.tensor(distances, dtype=torch.int)


def node_distance_(graph: Graph, start: int) -> Tensor:
    N = len(graph)
    visited = torch.zeros((N,), dtype=torch.bool)
    current = torch.zeros((N,), dtype=torch.bool)
    current[start] = True

    distance = torch.empty((N,), dtype=torch.int)
    distance[:] = -1

    for i in range(N):
        distance[current] = i
        if torch.all(visited == 1):
            break
        visited |= current
        current = torch.any(graph.adj * visited.unsqueeze(1), dim=0)
        current &= ~visited

    return distance


def comm_bound(comm: Comm, prob: Tensor) -> float:
    cum_dist = torch.zeros((1,))
    prob_srt, _ = torch.sort(prob, descending=True)
    for p, dist in zip(prob_srt, comm_dist_iterator(comm)):
        cum_dist += dist * p
    return cum_dist.item()


def dist_comm_bound(graph: Graph, start: int, comm: Comm, prob: Tensor) \
        -> float:
    bound, depth, width = 0.0, 0, 1
    distance = node_distance(graph, start)
    max_distance = torch.max(distance)
    depths = [[int(j.item()) for j in (distance==i).nonzero(as_tuple=True)[0]]
              for i in range(max_distance+1)]
    curr: list[int] = []
    while len(curr) != 0 or depth <= max_distance:
        if depth < len(depths):
            curr += depths[depth]
        if len(curr) <= width:
            assigned, curr = curr, []
        else:
            curr = sorted(curr, key=lambda i : prob[i], reverse=True)
            assigned, curr = curr[:width], curr[width:]
        p = sum([prob[i] for i in assigned], torch.zeros((1,))).item()
        bound += depth * p
        width *= comm[depth]
        depth += 1
    return bound


def dist_comm_bound_uniform(graph: Graph, start: int, comm: Comm) -> float:
    distance = node_distance(graph, start)
    sorted_distance = torch.sort(distance)[0]
    comm_distance = torch.tensor([i for _, i in 
            zip(range(len(graph)), comm_dist_iterator(comm))])
    max_distance = torch.maximum(sorted_distance, comm_distance)
    return torch.sum(max_distance).item() / len(graph)
