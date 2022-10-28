import unittest

import torch

from graph import Graph, create_grid
from lossless import (all_reachable, dag_paths_rev, edge_iterator, find_path,
                      find_unitary_flow, lossless_code, lossless_code_NP,
                      patch_paths, path_cover_from_adj, path_dag, set_cover,
                      transitive_closure_dag)


class LosslessTest(unittest.TestCase):
    def test_dag_1D(self):
        graph = create_grid([5])

        graph_dag = path_dag(graph, 1)

        exp_adj = torch.zeros((5, 5), dtype=torch.int)
        exp_adj[1,0] = 1   # 1 step
        exp_adj[1,2] = 1   # 1 step
        exp_adj[2,3] = 1   # 2 steps
        exp_adj[3,4] = 1   # 3 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))

    def test_dag_2D_1(self):
        grid = create_grid([3, 2])
        graph_dag = path_dag(grid, 1)

        exp_adj = torch.zeros((6, 6), dtype=torch.int)
        exp_adj[1,0] = 1   # 1 step
        exp_adj[1,4] = 1   # 1 step
        exp_adj[1,2] = 1   # 1 steps
        exp_adj[0,3] = 1   # 2 steps
        exp_adj[4,3] = 1   # 2 steps
        exp_adj[2,5] = 1   # 2 steps
        exp_adj[4,5] = 1   # 2 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))
        
    def test_dag_2D_2(self):
        graph = create_grid([3, 2])
        graph.adj[1,3] = 1
        graph.adj[3,1] = 1

        graph_dag = path_dag(graph, 0)

        exp_adj = torch.zeros((6, 6), dtype=torch.int)
        exp_adj[0,1] = 1   # 1 step
        exp_adj[0,3] = 1   # 1 step
        exp_adj[1,2] = 1   # 2 steps
        exp_adj[1,4] = 1   # 2 steps
        exp_adj[3,4] = 1   # 2 steps
        exp_adj[2,5] = 1   # 3 steps
        exp_adj[4,5] = 1   # 3 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))
    
    def test_dag_2D_end(self):
        graph = create_grid([3, 2])
        graph.adj[1,3] = 1
        graph.adj[3,1] = 1

        graph_dag = path_dag(graph, 0, 4)

        exp_adj = torch.zeros((6, 6), dtype=torch.int)
        exp_adj[0,1] = 1   # 1 step
        exp_adj[0,3] = 1   # 1 step
        exp_adj[1,2] = 1   # 2 steps
        exp_adj[1,4] = 1   # 2 steps
        exp_adj[3,4] = 1   # 2 steps

        self.assertTrue(torch.all(graph_dag.adj==exp_adj))
        
    def test_dag_paths_rev(self):
        graph = create_grid([3, 2])
        graph_dag = path_dag(graph, 0)

        paths = dag_paths_rev(graph_dag, 0)

        expected_paths = [
            [0], [0, 1], [0, 1, 2], [0, 1, 2, 5], 
            [0, 1, 4], [0, 1, 4, 5], [0, 3], [0, 3, 4], [0, 3, 4, 5]
        ]

        for path, exp_path in zip(paths, expected_paths):
            for n in path:
                self.assertTrue(n in exp_path)
            for n in exp_path:
                self.assertTrue(n in path)

    def test_path_cover(self):
        sets = [
            [2, 3],
            [0, 1, 2],
            [1, 2],
            [4]
        ]

        idx, cover = set_cover(sets, 5)

        exp_idx = [0, 1, 3]
        exp_cover = [[2, 3], [0, 1, 2], [4]]

        for a, b in zip(idx, exp_idx):
            self.assertTrue(a==b)

        for a, b in zip(cover, exp_cover):
            self.assertTrue(a==b)

    def test_lossless_code_NP(self):
        graph = create_grid([4, 4])
        paths = lossless_code_NP(graph, 5)
        self.assertTrue(len(paths)==6)
        uni = set()
        for path in paths:
            uni = uni.union(path)
        self.assertTrue(len(uni)==16)

    def test_find_path(self):
        graph = create_grid([5, 5])
        path = find_path(graph, 3, 6)
        exp_path = [3, 2, 1, 6]
        self.assertTrue(len(path)==len(exp_path))
        for node, exp_node in zip(path, exp_path):
            self.assertTrue(node==exp_node)\

    def test_find_root_path_disconnected(self):
        graph = create_grid([5])
        graph.adj[2,3]=0
        path = find_path(graph, 0, 4)
        self.assertTrue(len(path)==0)

    def test_flow(self):
        capacity = torch.zeros((7,7), dtype=torch.int)
        capacity[0,2] = 1
        capacity[0,3] = 1
        capacity[2,4] = 1
        capacity[3,4] = 1
        capacity[3,5] = 1
        capacity[4,1] = 1
        capacity[5,1] = 1
        capacity[6,1] = 1

        exp_flow = capacity.clone()
        exp_flow[3,4] = 0
        exp_flow[6,1] = 0
        exp_flow -= exp_flow.clone().T

        flow = find_unitary_flow(capacity, 0, 1)

        self.assertTrue(torch.all(flow==exp_flow))

    def test_flow_2(self):
        capacity = torch.zeros((10,10), dtype=torch.int)
        capacity[0,5] = 1
        capacity[0,6] = 1
        capacity[1,7] = 1
        capacity[2,7] = 1
        capacity[8,0:4] = 1
        capacity[4:8,9] = 1

        flow = find_unitary_flow(capacity, 8, 9)

        exp_flow = torch.zeros((10,10), dtype=torch.int)
        paths = [[8, 0, 5, 9], [8, 1, 7, 9]]
        for path in paths:
            for a, b in zip(path[:-1], path[1:]):
                exp_flow[a,b] = 1
                exp_flow[b,a] = -1

        self.assertTrue(torch.all(flow==exp_flow))

    def test_edge_iterator(self):
        adj = torch.zeros((10, 10), dtype=torch.int)
        edges = [(1,2), (3,2), (4,8), (6,1), (6,9), (8,0)]
        for i, j in edges:
            adj[i,j] = 1
        adj[3,3] = -1

        for (i, j), (i_exp, j_exp) in zip(edge_iterator(adj), edges):
            self.assertTrue(i == i_exp)
            self.assertTrue(j == j_exp)

    def test_path_cover_from_adj(self):
        adj = torch.zeros((4,4), dtype=torch.int)
        adj[0,1] = 1
        adj[1,3] = 1

        paths = path_cover_from_adj(adj)

        exp_paths = [[0,1,3], [2]]
        for path, exp_path in zip(paths, exp_paths):
            for n in path:
                self.assertTrue(n in exp_path)
            for n in exp_path:
                self.assertTrue(n in path)

    def test_lossless_code_1(self):
        graph = create_grid([2, 2])
        paths = lossless_code(graph, 0)
        
        exp_paths = [[0,1,3], [0,2]]
        for path, exp_path in zip(paths, exp_paths):
            self.assertTrue(len(path) == len(set(path)))
            for n in path:
                self.assertTrue(n in exp_path)
            for n in exp_path:
                self.assertTrue(n in path)

    def test_lossless_code_2(self):
        graph = create_grid([4, 4])
        paths = lossless_code(graph, 5)
        self.assertTrue(len(paths)==6)
    
    def test_lossless_code_3(self):
        adj = torch.zeros((8, 8), dtype=torch.int)
        edges = [(0,1), (0,2), (0,3), (1,4), (2,4), (3,4), (4,5), (4,6), (4,7)]
        for i, j in edges:
            adj[i,j] = 1
        graph = Graph(adj)

        paths = lossless_code(graph, 0)
        self.assertTrue(len(paths)==3)

        for path in paths:
            for a, b in zip(path[:-1], path[1:]):
                self.assertTrue(graph.adj[a,b]==1)

    def test_all_reachable(self):
        adj = torch.zeros((7, 7), dtype=torch.int)
        edges = [(0,1), (1,2), (2,3), (2,4), (4,5), (0,6), (6,4)]
        for i, j in edges:
            adj[i,j] = 1
        graph = Graph(adj)

        mem: list[torch.Tensor|None] = [None] * len(graph)
        reach, mem = all_reachable(graph, 0, mem)  

        exp_reach = torch.tensor([0, 1, 1, 1, 1, 1, 1], dtype=torch.int)
        self.assertTrue(torch.all(reach==exp_reach))

        self.assertTrue(len(mem)==7)
        exp_mem = [
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
        ]
        for m, exp_m in zip(mem, exp_mem):
            self.assertTrue(torch.all(m==torch.tensor(exp_m, dtype=torch.int)))

    def test_transitive_closure_dag(self):
        adj = torch.zeros((7, 7), dtype=torch.int)
        edges = [(0,1), (1,2), (2,3), (2,4), (4,5), (0,6), (6,4)]
        for i, j in edges:
            adj[i,j] = 1
        graph = Graph(adj)

        res = transitive_closure_dag(graph)

        exp_adj = torch.Tensor([
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
        ])
        self.assertTrue(torch.all(res.adj==exp_adj))

    def test_patch_paths(self):
        paths = [[0, 2, 4, 6]]
        adj = torch.zeros((7, 7), dtype=torch.int)
        for i in range(6):
            adj[i,i+1] = 1
        dag = Graph(adj+adj.T)
        new_paths = patch_paths(paths, dag)
        self.assertEqual(len(new_paths), 1)
        self.assertListEqual(new_paths[0], [0,1,2,3,4,5,6])
