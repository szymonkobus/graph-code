import unittest

import torch

from graph import create_grid
from lossless import dag_paths, lossless_code_NP, path_cover, shortest_path_dag


class LosslessTest(unittest.TestCase):
    def test_dag_1D(self):
        graph = create_grid([5])

        graph_dag = shortest_path_dag(graph, 1)

        exp_adj = torch.zeros((5, 5), dtype=int)
        exp_adj[1][0] = 1   # 1 step
        exp_adj[1][2] = 1   # 1 step
        exp_adj[2][3] = 1   # 2 steps
        exp_adj[3][4] = 1   # 3 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))

    def test_dag_2D_1(self):
        grid = create_grid([3, 2])
        graph_dag = shortest_path_dag(grid, 1)

        exp_adj = torch.zeros((6, 6), dtype=int)
        exp_adj[1][0] = 1   # 1 step
        exp_adj[1][4] = 1   # 1 step
        exp_adj[1][2] = 1   # 1 steps
        exp_adj[0][3] = 1   # 2 steps
        exp_adj[4][3] = 1   # 2 steps
        exp_adj[2][5] = 1   # 2 steps
        exp_adj[4][5] = 1   # 2 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))
        
    def test_dag_2D_2(self):
        graph = create_grid([3, 2])
        graph.adj[1][3] = 1
        graph.adj[3][1] = 1

        graph_dag = shortest_path_dag(graph, 0)

        exp_adj = torch.zeros((6, 6), dtype=int)
        exp_adj[0][1] = 1   # 1 step
        exp_adj[0][3] = 1   # 1 step
        exp_adj[1][2] = 1   # 2 steps
        exp_adj[1][4] = 1   # 2 steps
        exp_adj[3][4] = 1   # 2 steps
        exp_adj[2][5] = 1   # 3 steps
        exp_adj[4][5] = 1   # 3 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))
        
    def test_dag_paths(self):
        graph = create_grid([3, 2])
        graph_dag = shortest_path_dag(graph, 0)

        paths = dag_paths(graph_dag, 0)

        expected_paths = [
            {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 5}, 
            {0, 1, 4}, {0, 1, 4, 5}, {0, 3}, {0, 3, 4}, {0, 3, 4, 5}
        ]

        for path in expected_paths:
            self.assertTrue(path in paths)
        for path in paths:
            self.assertTrue(path in expected_paths)    
        
    def test_path_cover(self):
        sets = [
            {2, 3},
            {0, 1, 2},
            {1, 2},
            {4}
        ]

        idx, cover = path_cover(sets, 5)

        exp_idx = [0, 1, 3]
        exp_cover = [{2, 3}, {0, 1, 2}, {4}]

        for a, b in zip(idx, exp_idx):
            self.assertTrue(a==b)

        for a, b in zip(cover, exp_cover):
            self.assertTrue(a==b)

    def test_lossless_NP(self):
        graph = create_grid([2, 3])
        cover, set_cover = lossless_code_NP(graph, 0)
        self.assertTrue(len(cover)==2)
