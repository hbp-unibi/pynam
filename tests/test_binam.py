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

import StringIO
import numpy as np
import pynam.binam

class TestBinaryMatrix(unittest.TestCase):

    def test_set_set(self):
        # Load the BinaryMatrix from a list
        a = pynam.binam.BinaryMatrix()
        a.set([
                [0, 1, 0, 0, 1],
                [0, 0, 0, 1],
                [],
                [0, 1, 0]
            ])

        # Save the BinaryMatrix to a list and compare it to the expected output
        self.assertEqual(a.get(True), [
                [0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]
            ])

        # Load the BinaryMatrix from a numpy array
        a = pynam.binam.BinaryMatrix()
        arr = np.zeros((4, 5), dtype=np.uint8)
        arr[1, 4] = 1
        arr[2, 4] = 1
        arr[0, 3] = 1
        arr[3, 3] = 1
        a.set(arr)

        # Save the BinaryMatrix to a list and compare it to the expected output
        np.testing.assert_equal(a.get(), arr)

    def test_set_get_item(self):
        a = pynam.binam.BinaryMatrix()
        a.resize(10, 5)
        a[5, 2] = 1
        a[3, 1] = 1
        a[2, 4] = 1
        a[9, 0] = 1
        a[0, 0] = 1

        self.assertEqual(a[5, 2], 1)
        self.assertEqual(a[3, 1], 1)
        self.assertEqual(a[2, 4], 1)
        self.assertEqual(a[9, 0], 1)
        self.assertEqual(a[0, 0], 1)
        self.assertEqual(a[0, 1], 0)
        self.assertEqual(a[1, 0], 0)
        self.assertEqual(a[1, 1], 0)

    def test_shape(self):
        a = pynam.binam.BinaryMatrix()
        a.resize(10, 5)

        self.assertEqual(a.shape, (10, 5))
        self.assertEqual(len(a), 10)

    def test_row_col(self):
        a = pynam.binam.BinaryMatrix()
        a.resize(10, 5)

        a[5, 2] = 1
        a[3, 1] = 1
        a[2, 4] = 1
        a[9, 0] = 1
        a[0, 0] = 1

        np.testing.assert_equal(a[0], [1, 0, 0, 0, 0])
        np.testing.assert_equal(a[2], [0, 0, 0, 0, 1])
        np.testing.assert_equal(a[3], [0, 1, 0, 0, 0])
        np.testing.assert_equal(a[5], [0, 0, 1, 0, 0])
        np.testing.assert_equal(a[9], [1, 0, 0, 0, 0])
        np.testing.assert_equal(a[4], [0, 0, 0, 0, 0])

        np.testing.assert_equal(a.row(0), [1, 0, 0, 0, 0])
        np.testing.assert_equal(a.row(2), [0, 0, 0, 0, 1])
        np.testing.assert_equal(a.row(3), [0, 1, 0, 0, 0])
        np.testing.assert_equal(a.row(5), [0, 0, 1, 0, 0])
        np.testing.assert_equal(a.row(9), [1, 0, 0, 0, 0])
        np.testing.assert_equal(a.row(4), [0, 0, 0, 0, 0])

        np.testing.assert_equal(a.col(0), [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_equal(a.col(1), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(a.col(2), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_equal(a.col(4), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(a.col(5), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(a.row(0, True), [1, 0, 0, 0, 0])
        self.assertEqual(a.row(2, True), [0, 0, 0, 0, 1])
        self.assertEqual(a.row(3, True), [0, 1, 0, 0, 0])
        self.assertEqual(a.row(5, True), [0, 0, 1, 0, 0])
        self.assertEqual(a.row(9, True), [1, 0, 0, 0, 0])
        self.assertEqual(a.row(4, True), [0, 0, 0, 0, 0])

        self.assertEqual(a.col(0, True), [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(a.col(1, True), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(a.col(2, True), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(a.col(4, True), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(a.col(5, True), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class TestBiNAM(unittest.TestCase):

    def test_basic_functionality(self):
        # Create an empty BiNAM instance
        a = pynam.binam.BiNAM()
        self.assertEqual(0, a.n_in())
        self.assertEqual(0, a.n_out())

        # Resize the BiNAM to a 10 x 7 BiNAM
        a.resize(10, 7)
        self.assertEqual(10, a.n_in())
        self.assertEqual(7, a.n_out())

        # Train a few vector pairs
        a.train([0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 1])
        a.train([0, 0, 0, 1, 1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0])
        a.train([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0])

        # Make sure the internal matrix is correct
        out = StringIO.StringIO()
        a.serialize(out)
        self.assertEqual(out.getvalue(),
            "1011000\n1011000\n1011000\n1011111\n1010100\n" +
            "0001011\n1010100\n0001011\n0000000\n0000000\n")

        # Evaluate the BiNAM
        np.testing.assert_equal(a.evaluate([0, 0, 0, 1, 0, 1, 0, 1, 0, 0]),
            [0, 0, 0, 1, 0, 1, 1]);
        np.testing.assert_equal(a.evaluate([0, 0, 0, 1, 1, 0, 1, 0, 0, 0]),
            [1, 0, 1, 0, 1, 0, 0]);
        np.testing.assert_equal(a.evaluate([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            [1, 0, 1, 1, 0, 0, 0]);

    def test_train_matrix(self):
        mat_in = np.array([
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ])
        mat_out = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])

        binam = pynam.binam.BiNAM(6, 4)
        binam.train_matrix(mat_in, mat_out)
        np.testing.assert_equal([
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], binam.get())

    def test_evaluate_matrix(self):
        mat_in = np.array([
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ])
        mat_out = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ])

        binam = pynam.binam.BiNAM(6, 4)
        binam.train_matrix(mat_in, mat_out)
        mat_out_recall = binam.evaluate_matrix(mat_in)

        np.testing.assert_equal(mat_out_recall, mat_out)

