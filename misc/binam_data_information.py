#!/usr/bin/env python
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

import numpy as np
import scipy.io as sio
import multiprocessing

# Include the PyNAM folder
import sys
import os
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), "../pynam"))

import binam
import binam_data
import binam_utils

# Parameters
n_bits = 16
n_ones = 3
n_samples = 100
n_step = 1
n_it = 1000

# Calculated from the parameters
t = range(0, n_samples + n_step, n_step);
n_real_samples = len(t)

functions = {
    "r_dups":
        lambda: binam_data.generate_random(n_bits, n_ones, n_samples),
    "r_no_dups":
        lambda: binam_data.generate(n_bits, n_ones, n_samples, balance=False),
    "b_dups":
        lambda: binam_data.generate_naive(n_bits, n_ones, n_samples),
    "b_no_dups":
        lambda: binam_data.generate(n_bits, n_ones, n_samples)
}

# Calculate the occupancy of a BiNAM-Matrix
def calculate(func_in, title, auto=False, func_out=None):
    global results
    errs = np.zeros((n_it, n_real_samples), dtype=np.float32)
    info = np.zeros((n_it, n_real_samples), dtype=np.float32)

    f_in = functions[func_in]
    f_out = f_in if func_out == None else functions[func_out]

    for i in xrange(n_it):
        print title + " generation " + str(i + 1) + " of " + str(n_it)

        # Generate input and output data
        X = f_in()
        if auto:
            Y = X
        else:
            Y = f_out()

        # Train the BiNAM, calculate the error
        M = binam.BiNAM(n_bits, n_bits)
        for j in xrange(1, n_real_samples):
            for k in xrange(t[j - 1], t[j]):
                M.train(X[k], Y[k])
            localErrs = np.zeros(t[j], dtype=np.uint32)
            for k in xrange(t[j]):
                localErrs[k] = np.sum(M.evaluate(X[k]) - Y[k])
            errs[i, j] = np.sum(localErrs) / t[j]
            # TODO: This must be done differently for auto association
            info[i, j] = binam_utils.entropy_hetero(localErrs, n_bits, n_ones)
    return {
        "errs": errs,
        "info": info,
        "title": title,
        "params": {
            "func_in": str(func_in),
            "func_out": str(func_out),
            "title": title,
            "auto": auto
        }
    }

pool = multiprocessing.Pool(processes=None)
workers = [
    pool.apply_async(calculate, ("r_dups", "Random, with duplicates")), # 0
    pool.apply_async(calculate, ("r_no_dups", "Random, no duplicates")), # 1
    pool.apply_async(calculate, ("b_dups", "Balanced, with duplicates")), # 2
    pool.apply_async(calculate, ("b_no_dups", "Balanced, no duplicates")), # 3
    pool.apply_async(calculate, ("r_no_dups",
        "x: Random, no duplicates; y: Random, with duplicates",
        False, "r_dups")), # 4
    pool.apply_async(calculate, ("b_no_dups",
        "x: Balanced, no duplicates; y: Random, with duplicates",
        False, "r_dups")), # 5
    pool.apply_async(calculate, ("b_no_dups",
        "x: Balanced, no duplicates; y: Random, no duplicates",
        False, "r_no_dups")), # 6
    pool.apply_async(calculate, ("b_no_dups",
        "x: Balanced, no duplicates; y: Balanced, with duplicates",
        False, "b_dups")), # 7
    pool.apply_async(calculate, ("b_no_dups",
        "x: Balanced, no duplicates; y: Balanced, no duplicates",
        False, "b_no_dups")), # 8
    pool.apply_async(calculate, ("r_dups", "Random, with duplicates", True)), # 9
    pool.apply_async(calculate, ("r_no_dups", "Random, no duplicates", True)), # 10
    pool.apply_async(calculate, ("b_dups", "Balanced, with duplicates", True)), # 11
    pool.apply_async(calculate, ("b_no_dups", "Balanced, no duplicates", True)) # 12
]

# Fetch the results and reorganize them
results = {"t": t, "n_ones": n_ones, "n_bits": n_bits, "n_samples": n_samples}
for worker in workers:
    res = worker.get()
    for key in res:
        if not key in results:
            results[key] = []
        results[key].append(res[key])

sio.savemat("out/binam_data_information_" + str(n_bits) + "_" + str(n_ones)
        + "_" + str(n_samples) + "_" + str(n_it) + "_all.mat", results)

