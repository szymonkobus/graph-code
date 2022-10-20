import os
import random

import scipy.sparse
import torch

from graph import Graph
from lossless import Paths


class TreeWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.make_index()
        self._index_complement = None

    def make_index(self) -> None:
        self.index = []
        if os.path.exists(self.path):
            for file in os.listdir(self.path):
                if file[:5]=='tree_':
                    self.index.append((int(file[5:-4]), file))
        random.shuffle(self.index)

    def make_index_complement(self, size: int):        
        index_start = [i for i, _ in self.index]
        self.index_complement = [i for i in range(size) if i not in index_start]
        random.shuffle(self.index_complement)

    def count_trees(self) -> int:
        all_names = os.listdir(self.path)
        names = [name for name in all_names if name[:5]=='tree_']
        return len(names)

    def create(self, i: int, start: int) -> None:
        assert len(self.index)>=i
        entry = (start, self.tree_name(start))
        if len(self.index)==i:
            self.index.append(entry)
        else:
            self.index.append(self.index[i])
            self.index[i] = entry

    def load(self, i: int) -> tuple[Paths, int]:
        start, name = self.index[i]
        file_name = f'{self.path}/{name}'
        with open(file_name, 'r') as file:
            path_str = [ [int(i) for i in line.split(',')] for line in file]
        return path_str, int(start)
        
    def save(self, i: int, tree: Paths) -> None:
        file_name = f'{self.path}/{self.index[i][1]}'
        if not os.path.exists(file_name):
            with open(file_name, 'w') as file:
                for path in tree:
                    file.write(','.join([str(i) for i in path]) + '\n')

    def tree_name(self, start: int) -> str:
        return f'tree_{start}.txt'

    def force_start(self, start: int):
        self.index = [(i, name) for i, name in self.index if i==start]
        self.index_complement.append(start)


class GraphWriter:
    def __init__(self, path: str, conf) -> None:
        self.path = f'{path}/{self.graph_type_folder_name(conf)}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.make_index()
        self.minimnal = max([-1]+[j for _, j, _ in self.index])

    def make_index(self) -> None:
        self.index: list[tuple[int, int, str]] = []
        for file in os.listdir(self.path):
            if file[:6]=='graph_':
                tree_writer = TreeWriter(f'{self.path}/{file}')
                count =tree_writer.count_trees()
                self.index.append((count, int(file[6:]), file))
        random.shuffle(self.index)

    def group_index(self, treshold: int) -> None:
        less_tresh = [(c,i,name) for c,i,name in self.index if c <= treshold]
        more_tresh = [(c,i,name) for c,i,name in self.index if c > treshold]
        self.index = less_tresh + more_tresh

    def create(self, i: int) -> TreeWriter:
        assert len(self.index)>=i
        entry = (0,) + self.graph_folder_name(i)
        if len(self.index)==i:
            self.index.append(entry)
        else:
            self.index.append(self.index[i])
            self.index[i] = entry
        
        return TreeWriter(self.get_index_path(i))

    def load(self, i: int) -> tuple[Graph, TreeWriter]:
        graph_path = self.get_index_path(i)
        adj_name, sparse_name = self.graph_file_names()
        adj = torch.load(f'{graph_path}/{adj_name}')
        sparse_path = f'{graph_path}/{sparse_name}'
        adj_sparse = None
        if os.path.exists(sparse_path):
            adj_sparse = scipy.sparse.load_npz(sparse_path)
        graph = Graph(adj, adj_sparse=adj_sparse)
        tree_writer = TreeWriter(graph_path)
        return graph, tree_writer

    def save(self, i: int, graph: Graph) -> None:
        graph_folder = self.get_index_path(i)
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)
        adj_name, sparse_name = self.graph_file_names()
        graph_path = f'{graph_folder}/{adj_name}'
        if not os.path.exists(graph_path):
            torch.save(graph.adj, graph_path)
        sparse_path = f'{graph_folder}/{sparse_name}'
        if graph.has_sparse and not os.path.exists(sparse_path):
            scipy.sparse.save_npz(sparse_path, graph.adj_sparse)

    ### NAMES ###
    def get_index_path(self, i: int) -> str:
        _, _, name = self.index[i]
        return f'{self.path}/{name}'

    def graph_type_folder_name(self, conf) -> str:
        other = ''
        if conf.graph_type=='random':
            other = f'_A{conf.attachment_pow}'
        return f'G_{conf.graph_type}_D{conf.dim}{other}'

    def graph_folder_name(self, i: int) -> tuple[int, str]:
        # TODO: add checks to make sure it doesnt yet exist
        next = max(self.minimnal, i)
        self.minimnal = next + 1
        return next, f'graph_{next}'

    def graph_file_names(self) -> tuple[str,str]:
        return 'graph_adj.pt', 'graph_sparse.npz'


def get_writer(path, conf):
    return GraphWriter(path, conf)
