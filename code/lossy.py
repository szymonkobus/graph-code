from collections import defaultdict
from heapq import heapify, heappop, heappush
from itertools import chain
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


def join_nodes(parent_template: TNode, children: list[TNode]) -> TNode:
    if None in children:
        children = [child for child in children if child is not None]
    parent = TNode(parent_template.idx, children=children, 
                   name=parent_template.name)
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


def static_path_code_perf(paths: Paths, prob: Sequence[float], 
                          node_paths: Sequence[int], degree: int = 2) -> float:
    prob_paths = [0] * (len(paths)+1)
    for node, path in enumerate(node_paths):
        prob_paths[path] += prob[node]
    path_tree = node_code(prob_paths, len(paths), degree=degree)
    avg_n_code = expected_depth(path_tree, prob_paths)
    avg_n_move = distance_bound_paths(paths, node_paths, prob)
    return avg_n_move + (avg_n_code - 1)


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
    '''calculates expected distance to each vertex'''
    avg_distance = 0.
    for n_path, path in enumerate(paths):
        for depth, node in enumerate(path):
            if node_paths[node]==n_path:
                avg_distance += depth * prob[node]
    return avg_distance


def junction_code_graph(
        graph: Graph, start: int, prob: Sequence[float],
        path_prob: Callable[[Paths, int, Sequence[float]], Sequence[int]], 
        degree: int = 2) -> TNode:
    '''creates dynamic path coding tree for a graph'''
    paths = lossless_code(graph, start)
    tree = paths_to_tree(paths)
    node_paths = path_prob(paths, len(graph), prob)
    code_tree = junction_code(paths, prob, node_paths, degree, tree)
    return code_tree


def junction_code(paths: Paths, prob: Sequence[float], node_paths: Sequence[int],
                  degree: int = 2, tree: TNode | None = None) -> TNode:
    '''creates dynamic path coding tree'''
    if tree is None:
        tree = paths_to_tree(paths)
    numbered_paths = list(enumerate(paths))
    tree_prob = assign_path_prob(numbered_paths, tree, prob, node_paths)
    tree_wait = expand_junctions(tree_prob, degree=degree)
    return contract_junctions(tree_wait, paths, degree=degree)


def junction_code_perf(tree: TNode, prob: Sequence[float], 
                       node_paths: Sequence[int], degree: int = 2) -> float:
    '''calculates expected number of steps for junction coding'''
    pass


def assign_path_prob(numbered_paths: list[tuple[int, list[int]]], tree: TNode, 
                     prob: Sequence[float], node_paths: Sequence[int], 
                     depth: int = 1) -> TNode:
    '''
    adds probability and paths field to every node in the tree
    assumes each node belongs to at least 1 path
    '''
    groups = defaultdict(list)
    for i, path in numbered_paths:
        if len(path)>depth:
            groups[path[depth]].append((i, path))
    for child in tree.children:
        assign_path_prob(groups[child.idx], child, prob, node_paths, depth+1)

    tree.paths = [i for i, _ in numbered_paths]
    tree.p = sum([child.p for child in tree.children])
    if node_paths[tree.idx] in tree.paths:
        tree.p += prob[tree.idx]
    return tree


def expand_junctions(tree: TNode, degree: int = 2) -> TNode:
    '''expands each tree junction to be coded; assumes each node has field p'''
    tree.children = [expand_junctions(child, degree) for child in tree.children]
    if len(tree.children)>=2:
        join = lambda nodes: join_nodes_w_paths(tree, nodes)
        probs = [c.p for c in tree.children]
        tree = huffman(tree.children, probs, join, degree)
    return tree


def join_nodes_w_paths(parent_template: TNode, children: list[TNode]) -> TNode:
    parent = TNode(parent_template.idx, children=children, 
                   name=parent_template.name)
    parent.p = sum([child.p for child in children])
    parent.paths = list(chain(*[child.paths for child in children]))
    return parent


def contract_junctions(tree: TNode, paths: Paths, degree: int = 2, 
                       depth: int = -1) -> TNode:
    '''creats policy from tree path choices; looses name, p, and paths'''
    match tree.paths, tree.children:
        case ([path_no], []) | ([path_no], [_]) :
            depth += 1
            if len(paths[path_no])==depth+1:
                return TNode(paths[path_no][depth])
            child = contract_junctions(tree, paths, degree, depth)
            return TNode(paths[path_no][depth], children=[child])
        # case [path_no], [child]:
        #     depth += 1
        #     idx = paths[path_no][depth]
        #     if len(paths[path_no])==depth+1: # should be ==
        #         return TNode(idx)
        #     child = contract_junctions(tree, paths, degree, depth)
        #     return TNode(idx, children=[child])
        case _, [child]:
            return contract_junctions(child, paths, degree, depth)
        
    assert 1<len(tree.children) and len(tree.children)<=len(paths)
    idx = paths[tree.paths[0]][depth]
    if all([len(paths[path_no])>depth+1 for path_no in tree.paths]):
        next_idx = paths[tree.paths[0]][depth+1]
        if all([paths[path_no][depth+1]==next_idx for path_no in tree.paths]):
            idx = next_idx
            depth += 1
    children = [contract_junctions(child, paths, degree, depth)
                    for child in tree.children]
    return TNode(idx, children=children)


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
