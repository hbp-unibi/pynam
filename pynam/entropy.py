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
Contains utility functions used to calculate the entropy/storage capacity of
a BiNAM network.
"""

import numpy as np
import math

def ncr(n, k):
    """
    Implementation of the binomial coefficient (ncr) function for integer
    arguments. Returns an unlimited precision long integer. Returns zero for
    k > n.
    """
    if (k > n):
        return 0L
    if (k > n / 2):
        return ncr(n, n - k)
    res = 1
    for i in xrange(1, k + 1):
        res = (res * (n + 1 - i)) // i
    return res

def ncrr(x, y):
    """
    Implementation of the binomial coefficient (ncr) function for real arguments
    x and y (respectively corresponding to n and r).
    """
    return math.gamma(x+1.0) / (math.gamma(y+1.0) * math.gamma(x-y+1.0))

def lnncrr(x, y):
    """
    Returns the natural logarithm of the binomial coefficient for real
    arguments.
    """
    return (math.lgamma(x+1.0) - math.lgamma(y+1.0) - math.lgamma(x-y+1.0))

def expected_false_positives(n_samples, n_bits_out, n_ones_out, n_bits_in = 0,
        n_ones_in = 0):
    """
    Calculates the to be expected average number of false positives for the
    given data parameters.

    :param n_samples: number of trained samples.
    :param n_bits_out: number of output bits.
    :param n_ones_out: number of bits set to one in each output vector.
    :param n_bits_in: number of input bits. If smaller or equal to zero, the
    value will be copied from n_bits_out.
    :param n_ones_in: number of bits set to one in each output vector.If
    smaller or equal to zero, the value will be copied from n_bits_in.
    """
    if (n_bits_in <= 0):
        n_bits_in = n_bits_out
    if (n_ones_in <= 0):
        n_ones_in = n_ones_out
    N = n_samples
    p = float(n_ones_in * n_ones_out) / float(n_bits_in * n_bits_out)
    return ((n_bits_out - n_ones_out) *
            math.pow(1.0 - math.pow(1.0 - p, N), n_ones_in))

def expected_entropy(n_samples, n_bits_out, n_ones_out, n_bits_in = 0,
        n_ones_in = 0):
    """
    Calculates the expected entropy for data with the given parameters. See
    expected_false_positives for a description.
    """
    return entropy_hetero_uniform(expected_false_positives(n_samples,
        n_bits_out, n_ones_out, n_bits_in, n_ones_in), n_samples,
        n_bits_out, n_ones_out)

def entropy_hetero_uniform(err, n_samples, n_bits_out, n_ones_out):
    """
    Calculates the entropy for an estimated number of false positives err (which
    might be real-valued).

    :param errs: estimated number of false positives per sample.
    :param n_samples: number of samples.
    :param n_bits_out: number of output bits.
    :param n_ones_out: number of ones in the output.
    """
    v = 0
    for i in xrange(n_ones_out):
        v = v + math.log((n_bits_out - i) / (n_ones_out + err - i), 2.0)
    return n_samples * v

def entropy_hetero(errs, n_bits_out, n_ones_out):
    """
    Calculates the entropy from an errors-per sample matrix (returned by
    analyseSampleErrors) and for the given output vector size and the mean
    number of set bits. All values may also be real/floating point numbers,
    a corresponding real version of the underlying binomial coefficient is used.

    :param errs: errs is either an array of dictionaries containing "fn" and
    "fp"entries, where "fn" corresponds to the number of false negatives and
    "fp" to the number of false positives, or an array of numbers which
    correspond to the number of false positives.
    :params n_bits_out: length of the output vector.
    :params n_ones_out: number of ones in the output vector.
    """
    e = 0.0 # Entropy
    n_samples = len(errs)
    n = n_bits_out
    d = n_ones_out
    for t in xrange(n_samples):
        entry = errs[t]
        if isinstance(entry, dict):
            N0 = entry['fn'];
            N1 = entry['fp'];
            e += (lnncrr(n, d)
                    - lnncrr(N1 + d - N0, d - N0)
                    - lnncrr(n - N1 - d + N0, N0)) / math.log(2.0)
        else:
            for i in xrange(n_ones_out):
                e = e + math.log(float(n - i) / float(d + errs[t] - i), 2.0)
    return e

def calculate_errs(mat_out, mat_out_expected):
    """
    For each sample calculates the number of false negatives and false
    positives.
    """
    N, n = mat_out.shape
    res = [{'fn': 0, 'fp': 0} for _ in xrange(N)]
    for k in xrange(N):
        for j in xrange(n):
            if mat_out_expected[k, j] == 0:
                res[k]["fp"] = res[k]["fp"] + min(1, mat_out[k, j])
            else:
                res[k]["fn"] = res[k]["fn"] + 1 - min(1, mat_out[k, j])
    return res

def find_minimum_unimodal(f, a, b, tol=1):
    # Implementation of Golden section search
    # https://en.wikipedia.org/wiki/Golden_section_search
    gr = 0.5 * (math.sqrt(5)-1)
    c = b - gr * (b-a)
    d = a + gr * (b-a)
    while abs(c - d) > tol:
        fc = f(c)
        fd = f(d)
        if fc < fd:
            b = d
            d = c
            c = b - gr * (b-a)
        else:
            a = c
            c = d
            d = a + gr * (b-a)
    return (b + a) / 2

def optimal_sample_count(n_bits_in, n_bits_out, n_ones_in, n_ones_out):
    """
    Finds the sample count with the -- theoretically -- maximum information
    for the given data parameters.
    """
    I = 0.0
    p = 1.0 - float(n_ones_in * n_ones_out) / float(n_bits_in * n_bits_out)
    N_min = 0
    N_max = int(math.ceil(math.log(0.1) / math.log(p)))
    return int(find_minimum_unimodal(lambda N:
        -expected_entropy(int(N), n_bits_out, n_ones_out,
                n_bits_in, n_ones_in), N_min, N_max, tol=1))

def optimal_sample_count_naive(n_bits_in, n_bits_out, n_ones_in, n_ones_out):
    I = 0
    N = 0
    while True:
        N_next = N + 1
        I_next = expected_entropy(N_next, n_bits_out, n_ones_out, n_bits_in,
                n_ones_in)
        if (I_next < I):
            return N
        I = I_next
        N = N_next

def optimal_parameters(n_bits=100, n_samples=-1, n_bits_in=-1, n_bits_out=-1):
    n_min = 1
    n_max = (n_bits // 2) + 1
    n_bits_in = n_bits if n_bits_in <= 0 else n_bits_in
    n_bits_out = n_bits if n_bits_out <= 0 else n_bits_out

    arg = {
        "I_max": 0,
        "N_max": 0
    }

    def goal_fun(n_ones, arg):
        n_ones = int(n_ones)
        if (n_samples <= 0):
            N = optimal_sample_count(n_bits, n_bits, n_ones, n_ones)
        else:
            N = n_samples
        I = expected_entropy(N, n_bits, n_ones, n_bits, n_ones)
        if I == 0: # Quirk to make sure the function is truely unimodal
            I = -n_ones
        if I > arg["I_max"]:
            arg["I_max"] = I
            arg["N_max"] = N
        return -I

    n_ones = int(find_minimum_unimodal(lambda n: goal_fun(n, arg), n_min, n_max))

    return {
        "n_bits_in": n_bits,
        "n_ones_in": n_ones,
        "n_bits_out": n_bits,
        "n_ones_out": n_ones,
        "n_samples": arg["N_max"]
    }

def conventional_memory_entropy(n_bits_in, n_bits_out, n_ones_out):
    """
    Calculates storage capacity of a conventional MxN ROM holding data with the
    given specification.
    """
    return n_bits_in * lnncrr(n_bits_out, n_ones_out) / math.log(2.0)

