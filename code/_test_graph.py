import unittest

import torch

from graph import (Graph, create_grid, create_grid_skip, create_random_graph,
                   int_to_vec, vec_to_int)


class GraphTest(unittest.TestCase):
    def test_graph_size(self):
        adj = torch.tensor([[1, 0], [0, 0]])
        g = Graph(adj)
        self.assertTrue(len(g) == 2)

    def test_vec_to_int(self):
        ns = [0, 7, 11]
        vecs = [[0, 0, 0], [1, 0, 1], [2, 1, 1]]
        for exp_n, vec in zip(ns, vecs):
            base = torch.tensor([1, 3, 6, 12])
            n = vec_to_int(torch.tensor(vec), base)
            self.assertTrue(n==exp_n)

    def test_int_to_vec(self):
        ns = [0, 7, 11]
        vecs = [[0, 0, 0], [1, 0, 1], [2, 1, 1]]
        for n, exp_vec in zip(ns, vecs):
            base = torch.tensor([1, 3, 6, 12])
            vec = int_to_vec(n, base)
            self.assertTrue(torch.all(vec == torch.tensor(exp_vec)))

    def test_create_grid_1D(self):
        dim = [5]
        graph = create_grid(dim)

        exp_adj = torch.zeros((5, 5), dtype=torch.int)
        for i in range(4):
            exp_adj[i][i+1] = 1
            exp_adj[i+1][i] = 1

        self.assertTrue(torch.all(graph.adj==exp_adj))

    def test_create_grid_2D(self):
        dim = [3, 3]
        graph = create_grid(dim)

        exp_adj = torch.zeros((9, 9), dtype=torch.int)
        edge = [
            (0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
            (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)
        ]
        for i,j in edge:
            exp_adj[i][j] = 1
            exp_adj[j][i] = 1
 
        self.assertTrue(torch.all(graph.adj==exp_adj))

    def test_create_grid_3D(self):
        dim = [3, 2, 2]
        graph = create_grid(dim)

        neighbour_list = [(0,1), (1,2), (3,4), (4,5), (0,3), (1,4), (2,5)]
        exp_adj = torch.zeros((12, 12), dtype=torch.int)

        for i,j in neighbour_list:
            exp_adj[i][j] = 1
            exp_adj[j][i] = 1            
            exp_adj[i+6][j+6] = 1
            exp_adj[j+6][i+6] = 1
        
        for i in range(6):
            exp_adj[i][i+6] = 1
            exp_adj[i+6][i] = 1

        self.assertTrue(torch.all(graph.adj==exp_adj))

    def test_create_grid_skip_1(self):
        grid = create_grid_skip([2, 2], 0)
        exp_adj = torch.tensor([[0, 1, 1, 0],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 1, 0],])
        self.assertTrue(torch.all(grid.adj==exp_adj))

    def test_create_grid_skip_2(self):
        grid = create_grid_skip([2, 2], 2)
        exp_adj = torch.tensor([[0, 1, 1, 1],
                                [1, 0, 1, 1],
                                [1, 1, 0, 1],
                                [1, 1, 1, 0],])
        self.assertTrue(torch.all(grid.adj==exp_adj))
    
    def test_create_random_graph(self):
        for size in [2, 20, 200]:
            graph = create_random_graph(size, 1.)
            self.assertTrue(len(graph)==size)
            self.assertTrue(torch.all(graph.adj==graph.adj.T))
