import torch

from graph import Graph


def lossless_code_NP(graph: Graph, start: int):
    graph_dag = shortest_path_dag(graph, start)
    paths = dag_paths(graph_dag, start)
    assert len(paths) < 25
    cover, set_cover = path_cover(paths, len(graph))
    return cover, set_cover

def shortest_path_dag(graph: Graph, start: int) -> Graph:
    N = len(graph)
    adj_dag = torch.zeros((N, N), dtype=int)
    visited = torch.zeros((N,), dtype=bool)
    current = torch.zeros((N,), dtype=bool)
    current[start] = True

    for _ in range(N):
        current &= ~visited
        visited |= current
        adj_dag += graph.adj * current.unsqueeze(1) * ~visited.unsqueeze(0)
        current = torch.any(adj_dag * visited.unsqueeze(1), dim=0)

        if torch.all(visited == 1):
            break

    return Graph(adj_dag, graph.node_name)

def dag_paths(graph: Graph, node: int) -> list[set[int]]:
    r = [{node}]
    for next_node, b in enumerate(graph.adj[node]):
        if b==1:
            paths = dag_paths(graph, next_node)
            for path in paths:
                path.add(node)
            r += (paths)
    return r

def path_cover(sets: list[set[int]], N: int)->tuple[list[int],list[set[int]]]:
    sets_ohe = torch.zeros((len(sets), N), dtype=int)
    for i,s in enumerate(sets):
        for ele in s:
            sets_ohe[i][ele] = 1
   
    subsets_int = torch.arange(2**len(sets))
    mask = 2 ** torch.arange(len(sets)-1, -1, -1)
    subsets = subsets_int.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    
    is_cover = torch.all(torch.einsum('bi,sb->si',sets_ohe, subsets) != 0, dim=1)
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