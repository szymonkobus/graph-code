import unittest

from lossy import huffman, node_code, paths_to_tree


class LossyTest(unittest.TestCase):
    def test_node_code_bin(self):
        probs = [0.0, 0.5, 0.2, 0.3]
        res = node_code(probs, 0)
        exp_ids = \
            (0, [
                 1,
                (0, [3, 2])
            ])
        self.rec_id_check(res, exp_ids)

    def test_node_code_tri(self):
        probs = [0.0, 0.4, 0.2, 0.3, 0.1]
        res = node_code(probs, 0, degree=3)
        exp_ids = \
            (0, [
                1,
                (0, [2, 4]),
                3
            ])
        self.rec_id_check(res, exp_ids)

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
        exp_ids = \
        (0, [
            (1, [2]),
             3
        ])
        self.rec_id_check(res, exp_ids)

    def test_path_to_tree_2(self):
        paths = [[0,1,2], [0,1,3], [0,1,4]]
        res = paths_to_tree(paths)
        exp_ids = \
        (0, [
            (1, [2, 3, 4]),
        ])
        self.rec_id_check(res, exp_ids)

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
        exp_ids = \
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
        self.rec_id_check(res, exp_ids)

    def test_path_to_tree_4(self):
        paths = [[0,1,2,3], [0,4,2,5]]
        res = paths_to_tree(paths)
        exp_ids = \
        (0, [
            (1, [(2, [3])]),
            (4, [(2, [5])])
        ])
        self.rec_id_check(res, exp_ids)   

    def rec_id_check(self, tree, ids):
        if type(ids)==int:
           self.assertTrue(tree.id==ids)
        else:
            id, children_ids = ids
            self.assertTrue(tree.id==id)
            self.assertTrue(len(tree.children)==len(children_ids))
            for child, child_ids in zip(tree.children, children_ids):
                self.rec_id_check(child, child_ids)
