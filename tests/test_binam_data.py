# -*- coding: utf-8 -*-

#   PyNAM -- Python Neural Associative Memory Simulator and Evaluator
#   Copyright (C) 2015 Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest

import numpy as np
import numpy.testing
from pynam.binam_data import PermutationTrieNode, generate

class TestPermutationTrieNode(unittest.TestCase):
    def test_ctor(self):
        node = PermutationTrieNode(32, 3)
        self.assertEqual(node.idx, 32)
        self.assertEqual(node.parent, None)
        permutations = np.array([0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66,
                78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300,
                325, 351, 378, 406, 435, 465], dtype=np.uint64)
        numpy.testing.assert_equal(node.permutations, permutations)
        numpy.testing.assert_equal(node.max_permutations, permutations)

class TestGenerate(unittest.TestCase):

    def test_generate_all(self):
        res = generate(6, 3, 20)
        res = res[np.lexsort(res.T)]
        expected = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1]], dtype=np.uint8)
        numpy.testing.assert_equal(res, expected)

    def test_generate_all_multiple(self):
        res = generate(6, 3, 40)
        res = res[np.lexsort(res.T)]
        expected = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]], dtype=np.uint8)
        numpy.testing.assert_equal(res, expected)

    def test_generate_all_multiple_no_duplicates(self):
        res = generate(6, 3, 40, no_duplicates=True)
        res = res[np.lexsort(res.T)]
        expected = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1]], dtype=np.uint8)
        numpy.testing.assert_equal(res, expected)

    def test_generate_distribution(self):
        res = generate(6, 3, 20)
        for i in xrange(2, 22, 2):
            s = np.sum(res[0:i], 0)
            min_val = np.min(s)
            max_val = np.max(s)
            self.assertEqual(min_val, i // 2)
            self.assertEqual(max_val, i // 2)

    def test_seed(self):
        np.random.seed(123412845)
        a1 = np.random.randint(1000000)
        res1 = generate(128, 3, 100, no_duplicates=True, seed=1423452)
        a2 = np.random.randint(1000000)
        res2 = generate(128, 3, 100, no_duplicates=True, seed=1423452)
        a3 = np.random.randint(1000000)
        res3 = generate(128, 3, 100, no_duplicates=True)
        a4 = np.random.randint(1000000)
        self.assertTrue(np.all(res1 == res2))
        self.assertFalse(np.all(res1 == res3))
        self.assertFalse(a1 == a2 or a2 == a3 or a3 == a4)

