from collections import defaultdict
from heapq import heapify, heappop, heappush
from typing import Callable, Sequence, TypeVar

from graph import Graph
from lossless import Paths, lossless_code
from node import TNode

I = TypeVar('I')


def node_code(prob: Sequence[float], start: int, degree: int = 2) -> TNode:
    assert 0<=start and start<len(prob)
    root = TNode(start)
    if len(prob)==1:
        return root
    nodes = [TNode(i) for i in range(len(prob)) if i!=start]
    prob_nodes = [p for i, p in enumerate(prob) if i!=start]
    join = lambda nodes: join_nodes(root, nodes)
    code_tree = huffman(nodes, prob_nodes, join, degree=degree)
    return code_tree


def junction_code(graph: Graph, start: int, prob: Sequence[float],
                  degree: int = 2) -> TNode:
    paths = lossless_code(graph, start)
    tree = paths_to_tree(paths)
    code_tree = junction_code_tree(tree, prob, degree)
    return code_tree


def junction_code_tree(tree: TNode, prob: Sequence[float], degree: int = 2) \
        -> TNode:
    pass


def join_nodes(parent_template, children):
    if None in children:
        children = [child for child in children if child is not None]
    parent = TNode(parent_template.id, children=children, name=parent_template)
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
    

def paths_to_tree_rec(paths: Paths, id: int, depth: int) -> TNode:
    if len(paths)==1 and len(paths[0])==depth-1:
        return TNode(paths[0])

    groups = defaultdict(list)
    for path in paths:
        if len(path)>depth:
            groups[path[depth]].append(path)
    children = [paths_to_tree_rec(paths_group, id_group, depth+1) \
                for id_group, paths_group in groups.items()]
    node = TNode(id, children=children)
    return node
