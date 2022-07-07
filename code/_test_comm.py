import unittest

from comm import Comm, comm_dist_iterator


class CommTest(unittest.TestCase):
    def test_comm_iterator_1(self):
        comm = Comm(1)
        for i, j in zip(range(40), comm_dist_iterator(comm)):
            self.assertTrue(i == j)

    def test_comm_iterator_2(self):
        comm = Comm(2, 1)
        exp_comm_dist = [0, 1, 1, 2, 2, 2, 2, 3, 3]
        for i, j in zip(exp_comm_dist, comm_dist_iterator(comm)):
            self.assertTrue(i == j)

    def test_comm_iterator_3(self):
        comm = Comm(2, 2)
        exp_comm_dist = [0, 1, 1, 2, 2] + [3]*4 + [4]*4 + [5]*8 + [6]
        for i, j in zip(exp_comm_dist, comm_dist_iterator(comm)):
            self.assertTrue(i == j)

    def test_comm_iterator_4(self):
        comm = Comm(2, 3)
        exp_comm_dist = [0, 1, 1, 2, 2, 3, 3] + [4]*4
        for i, j in zip(exp_comm_dist, comm_dist_iterator(comm)):
            self.assertTrue(i == j)

    def test_comm_access(self):
        comm = Comm(3, 2)
        for i in range(10):
            self.assertTrue(comm[2*i] == 3)
            self.assertTrue(comm[2*i+1] == 1)