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

    def test_distance_bound_unifrom(self):
        graph = create_grid([3, 3])
        prob = torch.Tensor([1/9]).expand(9)
        bound = distance_bound(graph, 0, prob)
        exp_bound = 2
        self.assertTrue(bound == exp_bound)

    def test_distance_bound(self):
        graph = create_grid([3, 3])
        prob = torch.Tensor([0]*6 + [1/3]*3)
        bound = distance_bound(graph, 0, prob)
        exp_bound = 3
        self.assertTrue(bound == exp_bound)
    
    def test_communication_bound_unifrom(self):
        comm = Comm(2, 1)
        prob = torch.Tensor([1/9]).expand(9)
        bound = comm_bound(comm, prob)
        exp_bound = sum([0, 1, 1, 2, 2, 2, 2, 3, 3]) / 9
        self.assertTrue(abs(bound-exp_bound) < 1e-6)

    def test_communication_bound(self):
        comm = Comm(2, 1)
        prob = torch.Tensor([0, 2/3, 1/6, 1/6] + [0]*5)
        bound = comm_bound(comm, prob)
        exp_bound = 1/3
        self.assertTrue(abs(bound-exp_bound) < 1e-6)

    def test_distance_communication_bound_uniform(self):
        graph = create_grid([4, 4])
        comm = Comm(3, 1)
        prob = torch.Tensor([1/16]).expand(16)
        bound = dist_comm_bound(graph, 5, comm, prob)
        exp_bound = sum([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4])/16
        self.assertTrue(bound == exp_bound)

    def test_distance_communication_bound(self):
        graph = create_grid([4, 4])
        comm = Comm(3, 1)
        prob = torch.Tensor([1/16]*16)
        prob[1] -= 1/32
        prob[4] += 1/32
        bound = dist_comm_bound(graph, 5, comm, prob)
        exp_bound = sum([0, 1.5, 1, 1, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 4])/16
        self.assertTrue(bound == exp_bound)

    def test_distance_communication_bound_2(self):
        graph = create_grid([4, 4])
        comm = Comm(3, 2)
        prob = torch.Tensor([1/16]).expand(16)
        bound = dist_comm_bound(graph, 5, comm, prob)
        exp_bound = sum([0, 3*1, 3*2, 8*3, 1*4])/16
        self.assertTrue(bound == exp_bound)
