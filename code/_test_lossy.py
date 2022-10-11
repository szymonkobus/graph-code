import unittest

from node import TNode
from lossy import (huffman, node_code, paths_to_tree, first_paths_prob,
                   least_depths, expected_depth, static_path_code_perf,
                   distance_bound_paths)


class LossyTest(unittest.TestCase):
    def test_node_code_bin(self):
        probs = [0.0, 0.5, 0.2, 0.3]
        res = node_code(probs, 0)
        exp_idxs = \
            (0, [
                 1,
                (0, [3, 2])
            ])
        self.rec_idx_check(res, exp_idxs)

    def test_node_code_tri(self):
        probs = [0.0, 0.4, 0.2, 0.3, 0.1]
        res = node_code(probs, 0, degree=3)
        exp_idxs = \
            (0, [
                1,
                (0, [2, 4]),
                3
            ])
        self.rec_idx_check(res, exp_idxs)

    def test_static_path_code_perf(self):
        paths = [[0,1,2,3], [0,4,2,5], [0,6]]
        prob = [0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1]
        node_paths = [0, 0, 0, 0, 1, 1, 2]
        avg_steps = static_path_code_perf(paths, prob, node_paths)
        exp_avg_steps = 2.0
        self.assertTrue(abs(avg_steps-exp_avg_steps)<1e-8)

    def test_huffman_bin(self):
        items = ['a']
        probs = [1]
        merge = lambda *args: tuple(args)
        res = huffman(items, probs, merge)
        self.assertTrue(res == 'a')

    def test_huffman_bin_2(self):
        items = ['a', 'b', 'c', 'd']
        probs_unnorm = [7, 4, 2, 1]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]

        res = huffman(items, probs, tuple)

        exp = ('a', ('b', ('c', 'd')))
        self.assertTrue(res == exp)

    def test_huffman_bin_3(self):
        items = ['a', 'b', 'c', 'd', 'e']
        probs_unnorm = [15, 7, 6, 6, 5]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]

        res = huffman(items, probs, tuple, degree=2)

        exp = ((('b', 'c'), ('d', 'e')), 'a')
        self.assertTrue(res == exp)

    def test_huffman_tri(self):
        items = ['a', 'b', 'c', 'd', 'e']
        probs_unnorm = [15, 7, 6, 6, 5]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]

        res = huffman(items, probs, tuple, degree=3)

        exp = (('c', 'd', 'e'), 'a', 'b')
        self.assertTrue(res == exp)

    def test_huffman_tri_2(self):
        items = ['a', 'b', 'c', 'd']
        probs_unnorm = [15, 7, 6, 6]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]

        res = huffman(items, probs, tuple, degree=3)
        
        exp = ('a', ('c', 'd', None), 'b')
        self.assertTrue(res == exp)

    def test_path_to_tree_1(self):
        paths = [[0,1,2], [0,3]]
        res = paths_to_tree(paths)
        exp_idxs = \
        (0, [
            (1, [2]),
             3
        ])
        self.rec_idx_check(res, exp_idxs)

    def test_path_to_tree_2(self):
        paths = [[0,1,2], [0,1,3], [0,1,4]]
        res = paths_to_tree(paths)
        exp_idxs = \
        (0, [
            (1, [2, 3, 4]),
        ])
        self.rec_idx_check(res, exp_idxs)

    def test_path_to_tree_3(self):
        paths = [
            [43, 27, 15],
            [43, 27, 12, 7],
            [43, 27, 12, 5],
            # [43, 16, 7],
            [43, 16, 9, 5],
            [43, 16, 9, 4],
        ]
        res = paths_to_tree(paths)
        exp_idxs = \
        (43, [
            (27, [
                15,
                (12, [7, 5])
            ]),
            (16, [
                # 7,
                (9, [5, 4])
            ])
        ])
        self.rec_idx_check(res, exp_idxs)

    def test_path_to_tree_4(self):
        paths = [[0,1,2,3], [0,4,2,5]]
        res = paths_to_tree(paths)
        exp_idxs = \
        (0, [
            (1, [(2, [3])]),
            (4, [(2, [5])])
        ])
        self.rec_idx_check(res, exp_idxs)   

    def test_least_depth(self):
        t0_a = TNode(0, name='t0_a')
        t1_a = TNode(1, parent=t0_a, name='t1_a')
        t2   = TNode(2, parent=t1_a, name='t2')
        t0_b = TNode(0, parent=t1_a, name='t0_b')
        t0_c = TNode(0, parent=t0_a, name='t0_c')
        t1_b = TNode(1, parent=t0_c, name='t1_b')
        t3   = TNode(3, parent=t0_a, name='t3')
        depths = least_depths(t0_a, 4)
        exp_depths = [0, 1, 2, 1]
        self.assertTrue([a==b for a,b in zip(depths, exp_depths)])

    def test_expected_depth(self):
        t0 = TNode(0)
        t1 = TNode(1, parent=t0)
        t2 = TNode(2, parent=t1)
        t3 = TNode(3, parent=t2)
        t4 = TNode(4, parent=t2)
        prob = [0.1, 0.2, 0.3, 0.3, 0.1]
        avg_depth = expected_depth(t0, prob)
        exp_avg_depth = 2.0 
        self.assertTrue(avg_depth==exp_avg_depth)

    def test_distance_bound_paths(self):
        paths = [[0,1,2,5,8], [0,3,4,7], [0,3,6]]
        node_paths = [0,0,0,2,1,0,2,1,0]
        prob = [0.1]*9
        prob[1] = 0.2
        avg_distance = distance_bound_paths(paths, node_paths, prob)
        exp_avg_distance = 1.9
        self.assertTrue(abs(avg_distance-exp_avg_distance)<1e-8)

    def test_first_paths_prob(self):
        paths = [[0,1,2,3], [0,4,2,5]]
        res = first_paths_prob(paths, 6, [])
        exp_res = [0, 0, 0, 0, 1, 1]
        self.assertTrue(all([a==b for a,b in zip(res, exp_res)]))
        
    def rec_idx_check(self, tree, idxs):
        if type(idxs)==int:
           self.assertTrue(tree.idx==idxs)
        else:
            idx, children_idxs = idxs
            self.assertTrue(tree.idx==idx)
            self.assertTrue(len(tree.children)==len(children_idxs))
            for child, child_idxs in zip(tree.children, children_idxs):
                self.rec_idx_check(child, child_idxs)
