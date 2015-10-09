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
Contains the Network class which is responsible for translating the BiNAM into
a spiking neural network.
"""

import binam
import numpy as np
import pynnless as pynl
import pynnless.pynnless_utils as utils

class TopologyParameters(dict):
    """
    Contains parameters which concern the network topology -- the neuron type,
    the neuron multiplicity, neuron parameters and neuron parameter noise.
    """

    def __init__(self, data={}, params={}, param_noise={}, multiplicity=1,
            neuron_type=pynl.TYPE_IF_COND_EXP, weight=0.03, weight_stddev=0.0):
        utils.init_key(self, data, "params", params)
        utils.init_key(self, data, "param_noise", param_noise)
        utils.init_key(self, data, "multiplicity", multiplicity)
        utils.init_key(self, data, "neuron_type", neuron_type)
        utils.init_key(self, data, "weight", weight)
        utils.init_key(self, data, "weight_stddev", weight_stddev)

        self["params"] = pynl.PyNNLess.merge_default_parameters(
                data["params"], self["neuron_type"])

    def draw(self):
        res = dict(self["params"])
        for key in res.keys():
            if key in self["param_noise"]:
                res["key"] = np.random.normal(res["key"],
                        self["param_noise"][key])
        return pynl.PyNNLess.clamp_parameters(res)

    def draw_weight(self):
        if self["weight_stddev"] == 0.0:
            return self["weight"]
        return max(0.0, np.random.normal(self["weight"], self["weight_stddev"]))

class Network:

    # Number of samples
    N = 0

    # Input dimensionality
    m = 0

    # Output dimensionality
    n = 0

    # Input data matrix
    mat_in = None

    # Output data matrix
    mat_out = None

    # Internally cached BiNAM instance
    mem = None

    # Last sample until which the BiNAM has been trained
    last_k = 0

    def __init__(self, mat_in, mat_out, topology={}):
        """
        Constructor of the NetworkBuilder class -- the NetworkBuilder collects
        information about a network (storage matrix, noise parameters and input
        data).

        :param mat_in: Nxm matrix containing the input data
        :param mat_out: Nxn matrix containing the output data
        :param s: neuron multiplicity
        """
        self.mat_in = mat_in
        self.mat_out = mat_out

        self.topology = topology

        assert(mat_in.shape[0] == mat_out.shape[0])
        self.N = mat_in.shape[0]
        self.m = mat_in.shape[1]
        self.n = mat_out.shape[1]

    def in_coord(self, i, k=1):
        """
        Returns the coordinate of the input source with the given index.

        :param i: is the input component index.
        :param k: k'th neuron in the population corresponding to this index.
        """
        s = self.topology_params["multiplicity"]
        return (i * s + k, 0)

    def out_coord(self, i, k=1):
        """
        Returns the coordinate of the output neuron with the given index.

        :param i: is the output component index.
        :param k: k'th neuron in the population corresponding to this index.
        """
        s = self.topology_params["multiplicity"]
        return (self.m * s + j * s + k, 0)

    def build_topology(self, k=-1, seed=None):
        """
        Builds a network for a BiNAM that has been trained up to the k'th sample
        """

        # If k is smaller than zero, use the number of samples instead
        if k < 0:
            k = self.N

        # Train the BiNAM from the last trained sample last_k up to k
        if self.mem == None or self.last_k > k:
            self.mem = BiNAM(self.m, self.n)
        for l in xrange(self.last_k, k):
            self.mem.train(self.mat_in[l], self.mat_out[l])
        self.last_k = k

        # Build input and output neurons
        p = TopologyParameters(self.topology)
        s = p["multiplicity"]
        net = builder.Network()
        for i in xrange(self.m):
            for j in xrange(s):
                net.add_source()
        for i in xrange(self.n):
            for j in xrange(s):
                net.add_neuron(params=p.draw(), _type=p["neuron_type"])

        # Add all connections
        for i in xrange(self.m):
            for j in xrange(self.n):
                if net[i, j] != 0:
                    for k in xrange(s):
                        net.add_connection(in_coord(i, k), out_coord(j, k),
                                weight=p.draw_weight(), delay=0.0)
        return net
