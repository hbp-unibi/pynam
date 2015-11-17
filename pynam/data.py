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

"""
Contains a "generate" function which generates a set of training vectors that
can be used to train a BiNAM.
"""

import entropy
import utils
import numpy as np

#
# Classes
#

class PermutationTrieNode:
    """
    Class used internally by the "generate" method.

    The idea behind the permutation tree is the following: The tree stores
    sequences of bit indices that were set to one, the linear sequences are
    organized as a prefix tree (called "Trie", hence the name).

    The crucial idea is, that the sequences are sorted, from large to small
    indices. Thus a node can only link to children with smaller indices than
    the index of the parent node. This allows to calculate how many
    permutations can possibly start from this point. The permutation counter
    is decremented when ever a child (a new permutation) is added. This
    allows to prevent double permutations from being generated, as paths
    which have no permutations left, simply do not need to be persued any
    further.
    """

    def __init__(self, idx, remaining, parent = None):
        """
        Constructor of the PermutationTrieNode class.

        :param idx: index of this node within the parent node. As this node can
        only contain children with smaller indices, idx also corresponds to the
        length of the child list.
        :param remaining: remaining length of the sequence from this node on
        (number of bits that still have to be set).
        :param parent: parent node into which this node should be inserted/
        from which it should be fetched. If parent is None a new root node is
        generated.
        """
        self.idx = idx;
        max_val = np.iinfo(np.uint64).max
        self.max_permutations = np.fromiter((min(entropy.ncr(i, remaining - 1),
                max_val) for i in xrange(idx)), dtype=np.uint64, count=idx)
        self.permutations = self.max_permutations.copy()
        self.children = {}
        self.parent = parent
        self.remaining = remaining

    def fetch(self, idx):
        """
        Fetches or creates the child node with the given index.
        """
        if (not idx in self.children):
            self.children[idx] = PermutationTrieNode(
                    idx, self.remaining - 1, self)
        return self.children[idx]

    def decrement_permutation(self, idx):
        """
        Decrements the permutation counter for sequence continuing with "idx".
        """
        self.permutations[idx] = self.permutations[idx] - 1
        if (np.max(self.permutations) == 0):
            self.permutations = self.max_permutations.copy()
            return False
        return True

#
# Internal helper methods used for setting/restoring the random number generator
# state
#

def _initialize_generate(n_bits, n_ones, n_samples, seed=None):
    if (n_bits < 0 or n_ones < 0 or n_samples < 0):
        raise Exception("Arguments must be non-negative!")
    if (n_ones > n_bits):
        raise Exception("n_ones must be smaller or equal to n_bits!")
    return utils.initialize_seed(seed)

def _finalize_generate(old_random_state):
    utils.finalize_seed(old_random_state)

#
# Public methods
#

# Cache containing generated data for certain parameters
_generate_cache_ = {}

def generate(n_bits, n_ones, n_samples, abort_on_restart=False, seed=None,
        weight_choices=True, random=True, balance=True):
    """
    Generates a set of training vectors to be used in conjunction with the
    BiNAM. The returned data has the following properties:

    1. Bits are always allocated in a balanced fashion; if the first k returned
    vectors are summed, the value of the components will be on average
    (n_ones * k) / n_bits, with a maximum difference of 1 between the values.
    2. If "n_bits choose n_ones" samples are requested, all permutations of
    n_ones ones in n_bits bits will be returned. The first duplicate will be
    returned after all possible permutations have been returned. If the
    abort_on_restart flag is set, the function will abort and return a smaller
    than requested data matrix.
    3. The returned permutations are randomly shuffled and each sufficiently
    large block of samples will be uncorrelated.

    :param n_bits: is the size of the result vector.
    :param n_ones: specifies how many bits are set to one in the result vector.
    :param n_samples: number of samples to generate.
    :param abort_on_restart: if True, aborts once duplicates have to be generated.
    A samller data matrix than requested will be returned in this case.
    :param seed: If not "None", the random generator will be adjusted to use the
    given seed. The generator will be reset after this function ends.
    :param weight_choices: If True (default), the a correct weight is applied
    to the random choices, which is crucial to ensure the generated samples
    will be uncorrelated. Only has an effect if "random" is set to True.
    :param random: If False, a deterministic set of samples is generated.
    Default is True.
    :param balance: If False, does not perform balancing. Default is True.
    :return: a numpy ndarray containing the samples as rows.
    """

    # Try to read the generated data from the cache if it is supposed to be
    # generated deterministically
    global _generate_cache_
    if (not seed is None) or (not random):
        key = (n_bits, n_ones, n_samples, abort_on_restart, weight_choices,
                random, balance, seed)
        if key in _generate_cache_:
            return np.copy(_generate_cache_[key])

    old_random_state = _initialize_generate(n_bits, n_ones, n_samples, seed)
    try:
        res = np.zeros((n_samples, n_bits), dtype=np.uint8)
        usage = np.zeros(n_bits, dtype=np.uint32)
        root = PermutationTrieNode(n_bits, n_ones)
        for i in xrange(n_samples):
            node = root
            abort = False
            for j in xrange(n_ones):
                # Only select those paths which still have permutations left
                sel = node.permutations > 0

                if balance:
                    # From these select indices which balance the bit usage
                    usage_s = usage[:(node.idx)]
                    sel = np.logical_and(sel, usage_s == np.min(usage_s[sel]))

                    # Select indices which allow balancing after this layer
                    allowed = np.minimum(n_ones - j, np.cumsum(np.array(
                            usage_s == np.min(usage_s), dtype=np.uint16)))
                    best_sel = np.logical_and(sel, allowed == np.max(allowed))
                    if np.any(best_sel):
                        sel = best_sel

                # Weight the entries with the possible permutations the
                # corresponding path still can generate
                if random:
                    idcs = np.where(sel)[0]
                    if weight_choices:
                        ws = np.array(node.permutations[idcs], dtype=np.float64)
                        idx = np.random.choice(idcs, 1, p=ws/np.sum(ws))[0]
                    else:
                        idx = np.random.choice(idcs, 1)[0]
                else:
                    idx = node.idx - 1
                    while (not sel[idx]):
                        idx = idx - 1

                # Set the output bit, update the bit usage count
                res[i, idx] = 1
                usage[idx] = usage[idx] + 1

                # Abort if there are no more permutations left
                abort = (not node.decrement_permutation(idx) and node == root
                        and abort_on_restart) or abort

                # Descend into the tree
                node = node.fetch(idx)
            if abort:
                res.resize((i + 1, n_bits))
                return res
        return res
    finally:
        _finalize_generate(old_random_state)

        # Store the generated data in the cache
        if (not seed is None) or (not random):
            _generate_cache_[key] = np.copy(res)

def generate_naive(n_bits, n_ones, n_samples, seed=None):
    """
    Naive generation function which (in contrast to "generate") does ensure
    no duplicates are produced. Same parameters as above.
    """
    old_random_state = _initialize_generate(n_bits, n_ones, n_samples, seed)
    try:
        usage = np.zeros(n_bits, dtype=np.uint32)
        res = np.zeros((n_samples, n_bits), dtype=np.uint8)
        for i in xrange(n_samples):
            indices = []
            for j in xrange(n_ones):
                idx = np.random.choice(np.where(np.logical_and(
                    usage == np.min(usage), res[i] == 0))[0], 1)[0]
                usage[idx] = usage[idx] + 1
                res[i, idx] = 1
        return res
    finally:
        _finalize_generate(old_random_state)

def generate_random(n_bits, n_ones, n_samples, seed=None):
    """
    Random generation function which does not ensure balanced distribution of
    bits. Same paramters as above. Uses the Robert Floyd sampling algorithm.
    """
    old_random_state = _initialize_generate(n_bits, n_ones, n_samples, seed)
    try:
        res = np.zeros((n_samples, n_bits), dtype=np.uint8)
        for i in xrange(n_samples):
            for j in xrange(n_bits - n_ones, n_bits):
                idx = np.random.random_integers(0, j)
                if (res[i, idx] == 1):
                    res[i, j] = 1
                else:
                    res[i, idx] = 1
        return res
    finally:
        _finalize_generate(old_random_state)

#
# Main program
#
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print "./binam_data.py <BITS> <ONES> <SAMPLES>"
        sys.exit(1)
    data = generate(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    for row in data:
        print "".join(str(i) for i in row)

