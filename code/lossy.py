from collections import defaultdict
from heapq import heapify, heappop, heappush
from typing import Callable, Sequence, TypeVar

from graph import Graph
from lossless import Paths, lossless_code
from node import TNode

I = TypeVar('I')


def node_code(prob: Sequence[float], start: int, degree: int = 2) -> TNode:
    '''creates source code tree'''
    assert 0<=start and start<len(prob)
    root = TNode(start)
    if len(prob)==1:
        return root
    nodes = [TNode(i) for i in range(len(prob)) if i!=start]
    prob_nodes = [p for i, p in enumerate(prob) if i!=start]
    join = lambda children: join_nodes(root, children)
    code_tree = huffman(nodes, prob_nodes, join, degree=degree)
    return code_tree

def static_path_code_perf(paths: Paths, prob: Sequence[float], 
                          node_paths: Sequence[int], degree: int = 2) \
        -> float:
    prob_paths = [0] * (len(paths)+1)
    for node, path in enumerate(node_paths):
        prob_paths[path] += prob[node]
    path_tree = node_code(prob_paths, len(paths), degree=degree)
    avg_n_code = expected_depth(path_tree, prob_paths)
    avg_n_move = distance_bound_paths(paths, node_paths, prob)
    return avg_n_move + (avg_n_code - 1)

def junction_code_graph(graph: Graph, start: int, prob: Sequence[float],
                  node_paths: Sequence[int], degree: int = 2) -> TNode:
    '''creates dynamic path coding tree for a graph'''
    paths = lossless_code(graph, start)
    tree = paths_to_tree(paths)
    code_tree = junction_code(tree, prob, degree)
    return code_tree


def junction_code(tree: TNode, prob: Sequence[float], node_paths: Sequence[int],
                  degree: int = 2) -> TNode:
    '''creates dynamic path coding tree'''
    pass


def junction_code_perf(tree: TNode, prob: Sequence[float], 
                       node_paths: Sequence[int], degree: int = 2) -> float:
    '''calculates expected number of steps for junction coding'''
    pass


def join_nodes(parent_template: TNode, children: list[TNode]) -> TNode:
    if None in children:
        children = [child for child in children if child is not None]
    parent = TNode(parent_template.idx, children=children, name=parent_template)
    return parent


def huffman(items: list[I], prob: Sequence[float], join: Callable[..., I],
            degree: int = 2) -> I:
    heap = [(p, -i, item) for i, (item, p) in enumerate(zip(items, prob))]
    if degree != 2:
        n_fill = (1-len(items)) % (degree-1)
        fill = [(0., - i-len(items), None) for i in range(n_fill)]
        heap = fill + heap # type: ignore
    heapify(heap)
    while len(heap) >= degree:
        top = [heappop(heap) for _ in range(degree)]
        top_p, top_i, top_item = zip(*top)
        joined = (sum(top_p), max(top_i), join(list(reversed(top_item))))
        heappush(heap, joined)
    _, _, item = heap[0]
    return item


def paths_to_tree(paths: Paths) -> TNode:
    '''convert list of shortest paths to a tree'''
    return paths_to_tree_rec(paths, paths[0][0], 1)
    

def paths_to_tree_rec(paths: Paths, idx: int, depth: int) -> TNode:
    if len(paths)==1 and len(paths[0])==depth-1:
        return TNode(paths[0])

    groups = defaultdict(list)
    for path in paths:
        if len(path)>depth:
            groups[path[depth]].append(path)
    children = [paths_to_tree_rec(paths_group, id_group, depth+1) \
                for id_group, paths_group in groups.items()]
    node = TNode(idx, children=children)
    return node

    
def expected_depth(tree: TNode, prob: Sequence[float]) -> float:
    '''expected depth of nodes in the tree'''
    depths = least_depths(tree, len(prob))
    return sum([d*p for d,p in zip(depths, prob)])


def least_depths(tree: TNode, n: int) -> Sequence[int]:
    '''finds min depth for each node idx in the tree'''
    return least_depth_rec(tree, [None]*n, 0)


def least_depth_rec(tree: TNode, depths: Sequence[int], depth: int) \
        -> Sequence[int]:
    if (depths[tree.idx] is None) or (depth < depths[tree.idx]):
        depths[tree.idx] = depth
    for child in tree.children:
        depths = least_depth_rec(child, depths, depth+1)
    return depths


def distance_bound_paths(paths: Paths, node_paths: Sequence[int],
                         prob: Sequence[float]) -> float:
    '''calculates average expected distance to every node'''
    avg_distance = 0
    for n_path, path in enumerate(paths):
        for depth, node in enumerate(path):
            if node_paths[node]==n_path:
                avg_distance += depth * prob[node]
    return avg_distance


def first_paths_prob(paths: Paths, n_nodes: int, probs: Sequence[float]) \
        -> Sequence[int]:
    '''assigns each vertex to first path it is a part of'''
    assign = [None for _ in range(n_nodes)]
    for i, path in enumerate(paths):
        for node in path:
            if assign[node] is None:
                assign[node] = i
    assert not any([path is None for path in assign])
    return assign