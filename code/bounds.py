import torch
from torch import Tensor

from graph import Graph
from comm import Comm, comm_dist_iterator


def distance_bound(graph: Graph, start: int) -> float:
    distances = node_distance(graph, start)
    return torch.sum(distances).item() / len(graph)


def node_distance(graph: Graph, start: int) -> Tensor:
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


def comm_bound(comm: Comm, n_nodes: int) -> float:
    cum_dist = 0
    for _, dist in zip(range(n_nodes), comm_dist_iterator(comm)):
        cum_dist += dist
    return cum_dist / n_nodes


def dist_comm_bound(graph: Graph, start: int, comm: Comm) -> float:
    distance = node_distance(graph, start)
    sorted_distance = torch.sort(distance)[0]
    comm_distance = torch.tensor([i for _, i in 
            zip(range(len(graph)), comm_dist_iterator(comm))])
    max_distance = torch.maximum(sorted_distance, comm_distance)
    return torch.sum(max_distance).item() / len(graph)
