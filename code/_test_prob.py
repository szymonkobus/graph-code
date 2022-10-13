from math import sqrt
import unittest

import torch

from prob import first_paths_prob, polynomial_prob


class ProbTest(unittest.TestCase):
    def test_first_paths_prob(self):
        paths = [[0,1,2,3], [0,4,2,5]]
        res = first_paths_prob(paths, 6, [])
        exp_res = [0, 0, 0, 0, 1, 1]
        self.assertTrue(all([a==b for a,b in zip(res, exp_res)]))

    def test_polynomial_prob_1(self):
        p = polynomial_prob(4, 1.)
        exp_p = torch.tensor([4, 3, 2, 1], dtype=torch.float) / 10
        self.assertTrue(torch.all(p==exp_p))
    
    def test_polynomial_prob_2(self):
        p = polynomial_prob(4, 2.)
        exp_p = torch.tensor([16, 9, 4, 1], dtype=torch.float) / 30
        self.assertTrue(torch.all(p==exp_p))
    
    def test_polynomial_prob_3(self):
        p = polynomial_prob(4, 0.5)
        exp_p = torch.tensor([sqrt(4), sqrt(3), sqrt(2), 1], dtype=torch.float)
        exp_p /= torch.sum(exp_p)
        self.assertTrue(torch.all(p==exp_p))
