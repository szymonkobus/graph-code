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

    def test_comm_iterator_2(self):
        comm = Comm(2, 2)
        exp_comm_dist = [0, 1, 1, 2, 2] + [3] * 4 + [4] * 4 + [5] * 8 + [6]
        for i, j in zip(exp_comm_dist, comm_dist_iterator(comm)):
            self.assertTrue(i == j)
