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

"""
Contains an implementation of a classical BiNAM as well as methods to load and
save BiNAM matrices.
"""

import numpy as np

class BinaryMatrix:
    """
    Uses a dense numpy array to represent a matrix with binary values. For
    maximum efficiency the BiNAM is implemented internally as a compact array of
    64-bit integers.
    """

    # Used integer type
    int_type = np.uint64;

    # Width of the used integer type
    int_width = long(64)

    # Integer of the above type containing only ones
    int_ones = long(0xFFFFFFFFFFFFFFFF)

    # Array containing the actual matrix data
    arr = np.array([], dtype=int_type)

    # Number of rows (bits)
    n_rows = 0

    # Number of columns (bits)
    n_cols = 0

    # Number of columns in storage units
    n_cols_store = 0

    def __init__(self, rows=0, cols=0):
        """
        Constructor, creates a BinaryMatrix instance with the given size.
        """
        self.resize(rows, cols)

    def create_packed_bitvector(self, vec_in):
        """
        Converts the given input vector into packed row bit vector. The input
        vector vec_in must be a row or column vector. All entries in the input
        vector that do not equal to "0" are interpretet as "1".

        :param vec_in: is the input vector that should be converted to a packed
        row bitvector. Note that the input vector is forced to be a 1d-vector,
        so if a nxm matrix is given it will be reshaped to a vector of length
        n * m.
        """
        # Make sure the input is a vector
        vec_in = np.asarray(vec_in, dtype=np.uint8)
        vec_in = np.reshape(vec_in, (vec_in.size))
        n = vec_in.size
        n_store = (n + self.int_width - 1) / self.int_width

        # Fill the result
        res = np.zeros((n_store), dtype=self.int_type)
        idx = 0
        for i in xrange(n_store):
            cell = 0
            for j in xrange(self.int_width):
                if (idx >= n):
                    break
                if (vec_in[idx] != 0):
                    cell = cell | (1 << j)
                idx = idx + 1
            res[i] = cell
        return res

    def resize(self, rows, cols):
        """
        Resizes the binary matrix to a matrix with the given number of columns
        and rows.

        :param rows: number of rows in the resized matrix.
        :param cols: number of columns in the resized matrix.
        """
        self.n_rows = rows
        self.n_cols = cols
        self.n_cols_store = (cols + self.int_width - 1) / self.int_width
        self.arr = np.resize(self.arr, (self.n_rows, self.n_cols_store))

    def size(self):
        """
        Returns the size of the matrix as tuple consisting of the number of rows
        and the number of columns.
        """
        return (self.n_rows, self.n_cols)

    def __getitem__(self, tup):
        """
        Allows to access a single bit in the matrix, returns the value in the
        i-th row and the j-th colum. If a single value is given, returns the
        corresponding row.

        :param tup: tuple with the entries i and j, where i is the index of the
        row for which the bit should be written and j is the index of the column
        for which the bit should be written. If a single value i is given, only
        the row is returned.
        """
        if isinstance(tup, int):
            return self.row(tup)

        i, j = tup
        assert(i >= 0 and i < self.n_rows and j >= 0 and j < self.n_cols)
        return np.uint8(self.arr[i, j / self.int_width]
            & self.int_type(1 << (j % self.int_width)) > 0)

    def __setitem__(self, tup, val):
        """
        Allows to access a single bit in the matrix, returns the value in the
        i-th row and the j-th colum.

        :param tup: tuple with the entries i and j, where i is the index of the
        row for which the bit should be written and j is the index of the column
        for which the bit should be written.
        """
        i, j = tup
        assert(i >= 0 and i < self.n_rows and j >= 0 and j < self.n_cols)
        cell = self.arr[i, j / self.int_width]
        if (val == 0):
            cell = cell & ~self.int_type(1 << (j % self.int_width))
        else:
            cell = cell | self.int_type(1 << (j % self.int_width))
        self.arr[i, j / self.int_width] = cell

    def __len__(self):
        """
        Returns the number of rows in the matrix.
        """
        return self.n_rows

    @property
    def shape(self):
        """
        Returns the shape of the binary matrix.
        """
        return (self.n_rows, self.n_cols)

    def row(self, i, return_list=False):
        """
        Returns the i-th row stored in the matrix as a numpy uint8 vector.

        :param i: index of the row for which the row should be returned.
        """
        if (return_list == True):
            res = [np.uint8(0) for _ in xrange(self.n_cols)]
        else:
            res = np.zeros((self.n_cols), dtype=np.uint8)
        idx = 0
        for j in xrange(0, self.n_cols_store):
            val = self.arr[i, j]
            for b in xrange(0, self.int_width):
                if ((val & self.int_type(1 << b)) > 0):
                    res[idx] = 1
                idx = idx + 1
        return res

    def col(self, j, return_list=False):
        """
        Returns the j-th row stored in the matrix as a numpy uint8 vector.

        :param j: column of the binary matrix that should be returned.
        :param return_list: if True returns a python list instead of a numpy
        array.
        """
        if (return_list == True):
            res = [np.uint8(0) for _ in xrange(self.n_rows)]
        else:
            res = np.zeros((self.n_rows), dtype=np.uint8)
        col_idx = j / self.int_width
        col_bit = self.int_type(1 << (j % self.int_width))
        for i in xrange(0, self.n_rows):
            res[i] = np.uint8((self.arr[i, col_idx] & col_bit) > 0)
        return res

    def set(self, lst):
        """
        Loads the matrix content from a list of lists or a numpy array.

        :param lst: is the list of lists with which the array should be filled.
        """
        self.resize(len(lst), max(map(lambda l: len(l), lst)))
        for i in xrange(self.n_rows):
            idx = 0
            row = lst[i]
            for j in xrange(self.n_cols_store):
                cell = self.int_type(0)
                for k in xrange(self.int_width):
                    if (idx >= len(row)):
                        break
                    if (row[idx] != 0):
                        cell = cell | self.int_type(1 << k)
                    idx = idx + 1
                self.arr[i][j] = cell

    def get(self, return_list=False):
        """
        Returns the matrix content as a numpy array or a list of lists
        containing the array content.

        :param return_list: if True, returns a standard python list of lists
        instead of a numpy array.
        """
        if (return_list):
            res = [[np.uint8(0) for _ in xrange(self.n_cols)]
                    for _ in xrange(self.n_rows)]
        else:
            res = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        for i in xrange(self.n_rows):
            idx = 0
            for j in xrange(self.n_cols_store):
                for k in xrange(self.int_width):
                    if (idx >= self.n_cols):
                        break
                    if ((self.arr[i, j] & self.int_type(1 << k)) != 0):
                        res[i][idx] = np.uint8(1)
                    idx = idx + 1
        return res

    def deserialize(self, stream):
        """
        Reads a binary matrix from the given input stream. The input stream
        should be a text file consisting of "1" and "0" characters, with each
        line representing a row in the matrix.

        :param stream: input stream from which the matrix should be read.
        """
        lines = []
        for s in stream.readlines():
            line = []
            for c in s:
                if (c == '0' or c == '1'):
                    line.append(1 if c == '1' else 0)
            lines.append(line)
        self.set(lines)

    def serialize(self, stream):
        """
        Saves a binary matrix to the given output stream, writes each row as a
        sequence of "1" and "0" characters to a text file.

        :param stream: output stream to which the matrix should be written.
        """
        for i in xrange(self.n_rows):
            for j in xrange(self.n_cols_store):
                idx = 0
                for k in xrange(self.int_width):
                    if (idx >= self.n_cols):
                        break
                    if ((self.arr[i, j] & self.int_type(1 << k)) != 0):
                        stream.write("1")
                    else:
                        stream.write("0")
                    idx = idx + 1
            stream.write("\n")


class BiNAM(BinaryMatrix):
    """
    The BiNAM class implements evaluating and training a BiNAM matrix.
    """
    def n_in(self):
        """Returns the current number of input bits in the BiNAM (equals the
        number of rows in the storage matrix)."""
        return self.n_rows

    def n_out(self):
        """Returns the current number of output bits in the BiNAM (equals the
        number of columns in the storage matrix)."""
        return self.n_cols

    def train(self, vec_in, vec_out):
        """
        Trains the BiNAM matrix for the given input and output vector.

        :param vec_in: input vector. Its length must equal the current number of
        input bits.
        :param vec_out: output vector. Its length must equal the current number
        of output bits.
        """
        # Make sure the input/output arrays are numpy arrays and have the
        # correct size
        vec_in = np.asarray(vec_in, dtype=np.uint8)
        vec_out = np.asarray(vec_out, dtype=np.uint8)
        assert(vec_in.size == self.n_rows)
        assert(vec_out.size == self.n_cols)

        # Reshape the input vector to a 1d vector
        vec_in = np.reshape(vec_in, (self.n_rows))

        # Generate a bitwise row vector from the output vector
        vec_out = self.create_packed_bitvector(vec_out)

        # Perform the actual training by OR-ing vec_out to the rows selected by
        # vec_in
        for i in xrange(self.n_rows):
            if (vec_in[i] != 0):
                self.arr[i] = np.bitwise_or(self.arr[i], vec_out)

    def evaluate(self, vec_in, threshold = -1):
        """
        Returns the output of the BiNAM for the given input vector.

        :param vec_in: input vector that should be evaluated.
        :param threshold: threshold value -- values after the matrix-vector
        multiplication larger or equal to the threshold are set to one.
        """
        # Make sure vec_in is a numpy array and has the correct size
        vec_in = np.asarray(vec_in, dtype=np.uint8)
        assert(vec_in.size == self.n_rows)

        # Automatically select the threshold if none is given
        if (threshold < 0):
            threshold = np.sum(vec_in)

        # Create a single row containing the result of the vector/matrix
        # multiplication
        res = np.zeros((self.n_cols), dtype=np.uint32)
        for i in xrange(self.n_rows):
            if (vec_in[i] != 0):
                res = res + self.row(i)

        # Clamp according to the given threshold
        return np.asarray(res >= threshold, dtype=np.uint8)

