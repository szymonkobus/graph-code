import unittest

import torch

from graph import Graph
from lossy import (assign_path_prob, contract_junctions, distance_bound_paths,
                   expand_junctions, expected_depth, huffman, junction_code,
                   junction_code_graph, junction_code_perf, least_depths,
                   node_code, paths_to_tree, static_path_code_perf)
from node import TNode
from prob import first_paths_prob


class LossyTest(unittest.TestCase):
    def rec_idx_check(self, tree, idxs):
        if type(idxs)==int:
           self.assertTrue(tree.idx==idxs)
        else:
            idx, children_idxs = idxs
            self.assertTrue(tree.idx==idx)
            self.assertTrue(len(tree.children)==len(children_idxs))
            for child, child_idxs in zip(tree.children, children_idxs):
                self.rec_idx_check(child, child_idxs)

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

    def test_assign_prob(self):
        numbered_paths = [(0, [0, 1, 2]), (1, [0, 3])]
        v0 = paths_to_tree([path for _, path in numbered_paths])
        v1 = v0.children[0]
        v2 = v1.children[0]
        v3 = v0.children[1]
        prob = [0.2, 0.3, 0.1, 0.4]
        node_paths = [0, 0, 0, 1]
        assign_path_prob(numbered_paths, v0, prob, node_paths)
        self.assertEqual(v0.p, 1.)
        self.assertEqual(v1.p, 0.4)
        self.assertEqual(v2.p, 0.1)
        self.assertEqual(v3.p, 0.4)
        self.assertListEqual(v0.paths, [0, 1])
        self.assertListEqual(v1.paths, [0])        
        self.assertListEqual(v2.paths, [0])        
        self.assertListEqual(v3.paths, [1])
    
    def make_node(self, idx, parent, p, paths):
        node = TNode(idx, parent=parent)
        node.p = p
        node.paths = paths
        return node

    def test_code_junction(self):
        v0 = self.make_node(0, None, 1., [0,1,2,3,4])
        v1 = self.make_node(1, v0, 1/4, [0,1,2])
        v2 = self.make_node(2, v0, 1/4, [3])
        v3 = self.make_node(3, v0, 1/2, [4])
        v4 = self.make_node(4, v1, 1/4, [0,1,2])
        v5 = self.make_node(5, v4, 1/8, [0])
        v6 = self.make_node(6, v4, 1/16, [1])
        v3_b = self.make_node(3, v4, 0., [2])
        v7 = self.make_node(7, v3_b, 0., [2])

        tree = expand_junctions(v0, degree=2)

        exp_idxs = \
        (0, [
            (0, [
                (1, [
                    (4, [5,
                        (4, [
                            6, (3, [7])
                        ])
                    ])
                ]), 2
            ]), 3
        ])
        self.rec_idx_check(tree, exp_idxs)
        self.assertListEqual(tree.paths, [0,1,2,3,4])
        t0b, t3 = tree.children
        self.assertListEqual(t0b.paths, [0,1,2,3])
        self.assertListEqual(t3.paths, [4])
        t1, t2 = t0b.children
        self.assertListEqual(t1.paths, [0,1,2])
        self.assertListEqual(t2.paths, [3])
        t4, = t1.children
        self.assertListEqual(t4.paths, [0,1,2])
        t5, t4b = t4.children
        self.assertEqual(t5.paths, [0])
        self.assertEqual(t4b.paths, [1,2])
        t6, t3b = t4b.children
        self.assertEqual(t6.paths, [1])
        self.assertEqual(t3b.paths, [2])
        t7, = t3b.children
        self.assertListEqual(t7.paths, [2])

    def test_contract_junctions_1(self):
        paths = [[0,1,2],[0,1,3]]
        v0 = self.make_node(0, None, 0, [0,1])
        v1 = self.make_node(1, v0, 0, [0,1])
        v2 = self.make_node(2, v1, 0, [0])
        v3 = self.make_node(3, v1, 0, [1])
        tree = contract_junctions(v0, paths, 2)
        exp_idxs = (0, [(1, [2]), (1, [3])])
        self.rec_idx_check(tree, exp_idxs)
    
    def test_contract_junctions_2(self):
        v0 = self.make_node(0, None, 1., [0,1,2,3,4])
        v1 = self.make_node(1, v0, 1/4, [0,1,2])
        v2 = self.make_node(2, v0, 1/4, [3])
        v3 = self.make_node(3, v0, 1/2, [4])
        v4 = self.make_node(4, v1, 1/4, [0,1,2])
        v5 = self.make_node(5, v4, 1/8, [0])
        v6 = self.make_node(6, v4, 1/16, [1])
        v3_b = self.make_node(3, v4, 0., [2])
        v7 = self.make_node(7, v3_b, 0., [2])
        root = expand_junctions(v0, degree=2)
        
        paths = [[0,1,4,5],[0,1,4,6],[0,1,4,3,7],[0,2],[0,3]]
        tree = contract_junctions(root, paths, 2)

        exp_idxs = \
        (0, [
            (0, [
                (1, [
                    (4, [5]),
                    (4, [6, (3, [7])])
                ]), 2
            ]), 3
        ])
        self.rec_idx_check(tree, exp_idxs)

    def test_junction_code(self):
        paths = [[0,1,4,5],[0,1,4,6],[0,1,4,3,7],[0,2],[0,3]]
        prob = [0,0,1/4,1/2,1/16,2/16,1/16,0]
        node_paths = [0,0,3,4,0,0,1,2]
        tree = junction_code(paths, prob, node_paths, 2)
        exp_idxs = \
        (0, [
            (0, [
                (1, [
                    (4, [5]),
                    (4, [6, (3, [7])])
                ]), 2
            ]), 3
        ])
        self.rec_idx_check(tree, exp_idxs)

    def test_junction_code_perf(self):
        paths = [[0,1,4,5],[0,1,4,6],[0,1,4,3,7],[0,2],[0,3]]
        prob = [0,0,1/4,1/2,1/16,2/16,1/16,0.01]
        node_paths = [0,0,3,4,0,0,1,2]
        avg_perf = junction_code_perf(paths, prob, node_paths, 2)
        exp_perf = 31/16 + 0.05
        self.assertEqual(avg_perf, exp_perf)

    def test_junction_code_graph(self):
        adj = torch.zeros((9,9), dtype=torch.int)
        edges = [(0,1),(0,2),(0,3),(1,4),(4,5),(4,6),(4,8),(8,7),(8,1),(8,2),
                 (8,3),(7,5),(7,0),(5,2),(5,6),(6,5),(6,3),(4,1),(0,0),(1,2)]
        for i,j in edges:
            adj[i,j] = 1
        graph = Graph(adj)
        prob = [0,0,1/4,1/2,1/16,2/16,1/16,0,0]
        tree = junction_code_graph(graph, 0, prob, first_paths_prob, 2)
        exp_idxs = \
        (0, [
            (0, [
                (1, [
                    (4, [5]),
                    (4, [6, (8, [7])])
                ]), 2
            ]), 3
        ])
        self.rec_idx_check(tree, exp_idxs)
