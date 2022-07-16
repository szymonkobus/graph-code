import unittest

from lossy import huffman


class LosslessTest(unittest.TestCase):
    def test_huffman_1(self):
        items = ['a']
        probs = [1]
        merge = lambda *args: tuple(args)
        res = huffman(items, probs, merge)
        self.assertTrue(res == 'a')

    def test_huffman_2(self):
        items = ['a', 'b', 'c', 'd']
        probs_unnorm = [7, 4, 2, 1]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]
        merge = lambda *args: tuple(args)

        res = huffman(items, probs, merge)

        exp = ('a', (('d', 'c'), 'b'))
        self.assertTrue(res == exp)


    def test_huffman_3(self):
        items = ['a', 'b', 'c', 'd', 'e']
        probs_unnorm = [15, 7, 6, 6, 5]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]
        merge = lambda *args: tuple(args)

        res = huffman(items, probs, merge, degree=2)

        exp = ('a', (('e', 'c'), ('d', 'b')))
        self.assertTrue(res == exp)

    def test_huffman_4(self):
        items = ['a', 'b', 'c', 'd', 'e']
        probs_unnorm = [15, 7, 6, 6, 5]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]
        merge = lambda *args: tuple(args)

        res = huffman(items, probs, merge, degree=3)

        exp = ('b', 'a', ('e', 'c', 'd'))
        self.assertTrue(res == exp)

    def test_huffman_5(self):
        items = ['a', 'b', 'c', 'd']
        probs_unnorm = [15, 7, 6, 6]
        s = sum(probs_unnorm)
        probs = [p/s for p in probs_unnorm]
        merge = lambda *args: tuple(args)

        res = huffman(items, probs, merge, degree=3)
        
        exp = ('b', (None, 'c', 'd'), 'a')
        self.assertTrue(res == exp)
