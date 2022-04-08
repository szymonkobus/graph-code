import unittest
from code.graph import Graph

import torch


class GraphTest(unittest.TestCase):
    def test_graph_size(self):
        adj = torch.Tensor([[1, 0], [0, 0]])
        g = Graph(adj)
        self.assertTrue(len(g) == 2)