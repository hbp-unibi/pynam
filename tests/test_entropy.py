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
from pynam.entropy import ncr, entropy_hetero, entropy_hetero_uniform,\
        expected_false_positives, calculate_errs

class TestUtils(unittest.TestCase):

    def test_ncr(self):
        self.assertEqual(0L, ncr(3, 10))
        self.assertEqual(1L, ncr(0, 0))
        self.assertEqual([1L, 1L],
            [ncr(1, i) for i in xrange(2)])
        self.assertEqual([1L, 2L, 1L],
            [ncr(2, i) for i in xrange(3)])
        self.assertEqual([1L, 3L, 3L, 1L],
            [ncr(3, i) for i in xrange(4)])
        self.assertEqual([1L, 4L, 6L, 4L, 1L],
            [ncr(4, i) for i in xrange(5)])
        self.assertEqual([1L, 5L, 10L, 10L, 5L, 1L],
            [ncr(5, i) for i in xrange(6)])
        self.assertEqual([1L, 6L, 15L, 20L, 15L, 6L, 1L],
            [ncr(6, i) for i in xrange(7)])
        self.assertEqual([1L, 7L, 21L, 35L, 35L, 21L, 7L, 1L],
            [ncr(7, i) for i in xrange(8)])

    def test_expected_false_positives(self):
        N = 10
        c = 2
        d = 3
        m = 10
        n = 6
        self.assertAlmostEqual(expected_false_positives(N, n, d, m, c),
                (n - d) * 0.424219774)
        self.assertAlmostEqual(expected_false_positives(N, n, d),
                (n - d) * 0.84039451)

    def test_entropy_hetero(self):
        n_out_ones = 3
        n_out_bits = 16
        errs = [
            {
                "fp": 1,
                "fn": 0
            },
            {
                "fp": 0,
                "fn": 0
            },
            {
                "fp": 2,
                "fn": 0
            }
        ]
        errs2 = [1, 0, 2]
        v1 = entropy_hetero(errs, n_out_bits, n_out_ones)
        v2 = entropy_hetero(errs2, n_out_bits, n_out_ones)

        self.assertAlmostEqual(v1, 22.06592095594754)
        self.assertAlmostEqual(v2, 22.06592095594754)

    def test_entropy_hetero_uniform(self):
        n_samples = 10
        n_out_ones = 3
        n_out_bits = 16
        err = 1.5
        errs = [err for _ in xrange(n_samples)]
        errs2 = [{"fp": err, "fn": 0} for _ in xrange(n_samples)]

        v1 = entropy_hetero_uniform(err, n_samples, n_out_bits, n_out_ones)
        v2 = entropy_hetero(errs, n_out_bits, n_out_ones)
        v3 = entropy_hetero(errs2, n_out_bits, n_out_ones)
        self.assertAlmostEqual(v1, v2)
        self.assertAlmostEqual(v1, v3)

    def test_calculate_errs(self):
        mat_out_expected = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])
        mat_out = np.array([
            [1, 0.25, 0.2, 0],
            [1.5, 1.2, 0.25, 1],
            [0.1, 1, 0, 1],
        ])
        errs = calculate_errs(mat_out, mat_out_expected)
        self.assertAlmostEqual([{'fp': 0.25, 'fn': 0.8},
                {'fp': 1.25, 'fn': 0}, {'fp': 0.1, 'fn': 0}], errs)
