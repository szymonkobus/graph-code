import unittest

import torch

from bounds import comm_bound, dist_comm_bound, distance_bound, node_distance
from comm import Comm
from graph import create_grid


class BoundsTest(unittest.TestCase):
    def test_distance_1(self):
        graph = create_grid([20])
        distance = node_distance(graph, 3)
        exp_distance = torch.empty((20,), dtype=int)                
        exp_distance[0:4] = torch.tensor([3,2,1,0])
        exp_distance[3:] = torch.arange(17)
        self.assertTrue(torch.all(distance == exp_distance))

    def test_distance_2(self):
        # [0, 1, 2]
        # [1, 2, 3]
        # [2, 3, 4]
        graph = create_grid([3, 3])
        distance = node_distance(graph, 0)
        exp_distance = torch.tensor([0, 1, 2, 1, 2, 3, 2, 3, 4])                
        self.assertTrue(torch.all(distance == exp_distance))
    
    def test_distance_disjoint(self):
        graph = create_grid([5])
        graph.adj[2,3] = 0
        distance = node_distance(graph, 0)
        exp_distance = torch.tensor([0,1,2,-1,-1])
        self.assertTrue(torch.all(distance == exp_distance))

    def test_distance_bound(self):
        graph = create_grid([3, 3])
        bound = distance_bound(graph, 0)
        exp_bound = 2
        self.assertTrue(bound == exp_bound)
    
    def test_communication_bound(self):
        comm = Comm(2, 1)
        bound = comm_bound(comm, 9)
        exp_bound = sum([0, 1, 1, 2, 2, 2, 2, 3, 3]) / 9
        self.assertTrue(bound == exp_bound)

    def test_distance_communication_bound(self):
        graph = create_grid([4, 4])
        comm = Comm(3, 1)
        bound = dist_comm_bound(graph, 5, comm)
        exp_bound = sum([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]) / 16
        self.assertTrue(bound == exp_bound)
