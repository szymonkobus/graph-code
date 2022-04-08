import torch


class Graph:
    def __init__(self, adj, node_name=None):
        self.adj = adj
        self.node_name = node_name

    def __len__(self):
        return self.adj.shape[0]