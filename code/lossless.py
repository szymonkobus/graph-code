from collections import deque
from typing import Generator

import torch
from torch import Tensor

from graph import Graph

Paths = list[list[int]]


def lossless_code_NP(graph: Graph, start: int) -> Paths:
    graph_dag = path_dag(graph, start)
    paths = dag_covering_paths(graph_dag, start)
    assert len(paths) < 25
    _, paths = set_cover(paths, len(graph))
    return paths

def lossless_code(graph: Graph, start: int) -> Paths:
    #   n-out    n-in   source  sink 
    # [0,N-1], [N,2N-1], [2N], [2N+1] 
    graph_dag = path_dag(graph, start)
    N = len(graph)
    source, sink = 2*N, 2*N + 1
    capacity = torch.zeros((2*N + 2, 2*N + 2), dtype=int)
    capacity[source,:N] = 1
    capacity[:N,N:2*N] = graph_dag.adj
    capacity[N:2*N,sink] = 1

    flow = find_unitary_flow(capacity, source, sink)

    path_adj = flow[:N,N:2*N]
    paths = path_cover_from_adj(path_adj)

    full_paths = extend_paths(paths, graph_dag, start)

    return full_paths

def find_unitary_flow(capacity: Tensor, source: int, sink: int):
    M = len(capacity)
    flow = torch.zeros((M, M), dtype=int)

    for _ in range(M):
        residual_capacity = capacity - flow
        path = find_path(Graph(residual_capacity), source, sink)

        if len(path) == 0:
            break

        for beg, end in zip(path[:-1],path[1:]):
            flow[beg,end] += 1
            flow[end,beg] -= 1

    return flow

def find_path(graph: Graph, start: int, end: int) -> list[int]:
    graph_dag = path_dag(graph, start, end)
    if graph_dag is None:
        return []
    path = [end]
    node = end
    for _ in range(len(graph)):
        if node == start:
            return [node for node in reversed(path)]
        node = torch.argmax(graph_dag.adj[:,node]).item()
        path.append(node)
        
    return []

def find_path_dag(dag: Graph, start: int, end: int) -> list[int]:
    path = [end]
    node = end
    for _ in range(len(dag)):
        if node == start:
            return [node for node in reversed(path)]
        node = torch.argmax(dag.adj[:,node]).item()
        path.append(node)
    return []

def path_dag(graph: Graph, start: int, end: int=-1) -> Graph | None:
    N = len(graph)
    adj_dag = torch.zeros((N, N), dtype=int)
    visited = torch.zeros((N,), dtype=bool)
    current = torch.zeros((N,), dtype=bool)
    current[start] = True

    for _ in range(N):
        visited |= current
        adj_dag += graph.adj * current.unsqueeze(1) * ~visited.unsqueeze(0)
        current = torch.any(adj_dag * visited.unsqueeze(1), dim=0)
        current &= ~visited

        if torch.all(visited == 1) or (end!=-1 and current[end]):
            return Graph(adj_dag, graph.node_name)

    return None

def path_cover_from_adj(adj: Tensor) -> Paths:
    beg, end = {}, {}

    for i, j in edge_iterator(adj):
        match (i in end, j in beg):
            case False, False:
                path = deque([i, j])
                beg[i] = path
                end[j] = path
            case True, False:
                path = end.pop(i)
                path.append(j)
                end[j] = path
            case False, True:
                path = beg.pop(j)
                path.appendleft(i)
                beg[i] = path
            case True, True:
                left_path = end.pop(i)
                right_path = beg.pop(j)
                left_path += right_path
                end[left_path[-1]] = left_path

    paths = [list(path) for path in beg.values()]
    
    covered = sum(len(path) for path in paths)
    if covered != len(adj):
        node_covered = torch.zeros((len(adj),), dtype=bool)
        for path in paths:
            for n in path:
                node_covered[n] = True
        
        for i, n in enumerate(node_covered):
            if not n:
                paths.append([i])
    
    return paths

def edge_iterator(adj: Tensor) -> Generator[tuple[int, int], None, None]:
    for i, edges in enumerate(adj):
        for j, has_flow in enumerate(edges):
            if has_flow == 1:
                yield i, j

def extend_paths(paths: Paths, dag: Graph, start: int) -> Paths:
    full_paths = [find_path_dag(dag, start, path[0])[:-1] + path \
                    for path in paths]
    return full_paths

def extend_paths_NP(paths: Paths, dag: Graph, start: int) -> Paths:
    origin_paths: dict[int,list[int]] = {start : []}
    for path in dag_covering_paths(dag, start):
        for i, node in enumerate(path):
            origin_paths[node] = path[:i]

    full_paths = [origin_paths[path[0]] + path for path in paths]
    return full_paths

def dag_paths_rev(graph: Graph, node: int) -> Paths:
    r = [[node]]
    for next_node, b in enumerate(graph.adj[node]):
        if b==1:
            paths = dag_paths_rev(graph, next_node)
            for path in paths:
                path.append(node)
            r += paths
    return r

def dag_covering_paths(graph: Graph, root: int) -> Paths:
    paths = dag_covering_paths_rev(graph, root)
    return [[n for n in reversed(path)] for path in paths]

def dag_covering_paths_rev(graph: Graph, node: int) -> Paths:
    r = []
    for next_node, b in enumerate(graph.adj[node]):
        if b==1:
            paths = dag_covering_paths_rev(graph, next_node)
            for path in paths:
                path.append(node)
            r += paths
    if len(r) == 0:
        return [[node]]
    return r

def set_cover(sets: Paths, N: int) -> tuple[list[int],Paths]:
    sets_ohe = torch.zeros((len(sets), N), dtype=int)
    for i,s in enumerate(sets):
        for ele in s:
            sets_ohe[i][ele] = 1
   
    subsets_int = torch.arange(2**len(sets))
    mask = 2 ** torch.arange(len(sets)-1, -1, -1)
    subsets = subsets_int.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    
    is_cover = torch.all(torch.einsum('bi,sb->si',sets_ohe, subsets)!=0, dim=1)
    count = torch.sum(subsets, dim=1)
    
    idx, min_count = -1, N
    for i, (covers, cnt) in enumerate(zip(is_cover, count)):
        if covers and cnt < min_count:
            idx = i
            min_count = cnt

    if idx == -1:
        return [], []
        
    res_subset = subsets[idx]
    cover = []
    set_cover = []
    for i, covers in enumerate(res_subset):
        if covers:
            cover.append(i)
            set_cover.append(sets[i])

    return cover, set_cover
