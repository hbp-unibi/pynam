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
import bisect
import itertools
import numpy as np
import pynnless as pynl
import pynnless.pynnless_utils as utils

class InputParameters(dict):
    """
    Contains parameters which concern the generation of input data.
    """

    def __init__(self, data={}, burst_size=1, time_window=100.0, isi=1.0,
            sigma_t=0.0, sigma_t_offs=0.0):
        utils.init_key(self, data, "burst_size", burst_size)
        utils.init_key(self, data, "time_window", time_window)
        utils.init_key(self, data, "isi", isi)
        utils.init_key(self, data, "sigma_t", sigma_t)
        utils.init_key(self, data, "sigma_t_offs", sigma_t_offs)

    def build_spike_train(self, offs=0.0):
        res = np.zeros(self["burst_size"])

        # Draw the actual spike offset
        if (self["sigma_t_offs"] > 0):
            offs = np.random.normal(offs, self["sigma_t_offs"])

        # Calculate the time of each spike
        for i in xrange(self["burst_size"]):
            jitter = 0
            if (self["sigma_t"] > 0):
                jitter = np.random.normal(0, self["sigma_t"])
            res[i] = offs + i * self["isi"] + jitter
        return np.sort(res)


class TopologyParameters(dict):
    """
    Contains parameters which concern the network topology -- the neuron type,
    the neuron multiplicity, neuron parameters and neuron parameter noise.
    """

    def __init__(self, data={}, params={}, param_noise={}, multiplicity=1,
            neuron_type=pynl.TYPE_IF_COND_EXP, w=0.03, sigma_w=0.0):
        """
        Constructor of the TopologyParameters class.

        :param data: is a dictionary from which the arguments are copied
        preferably.
        :param params: dictionary containing the neuron parameters.
        :param param_noise: dictionary potentially containing a standard
        deviation for each parameter.
        :param multiplicity: number of neurons and signals each component is
        represented with.
        :param neuron_type: PyNNLess neuron type name.
        :param w: synapse weight
        :param sigma_w: synapse weight standard deviation.
        """
        utils.init_key(self, data, "params", params)
        utils.init_key(self, data, "param_noise", param_noise)
        utils.init_key(self, data, "multiplicity", multiplicity)
        utils.init_key(self, data, "neuron_type", neuron_type)
        utils.init_key(self, data, "w", w)
        utils.init_key(self, data, "sigma_w", sigma_w)

        self["params"] = pynl.PyNNLess.merge_default_parameters(
                self["params"], self["neuron_type"])

    def draw(self):
        res = dict(self["params"])
        for key in res.keys():
            if key in self["param_noise"] and self["param_noise"][key] > 0:
                res["key"] = np.random.normal(res["key"],
                        self["param_noise"][key])
        return pynl.PyNNLess.clamp_parameters(res)

    def draw_weight(self):
        if self["sigma_w"] <= 0.0:
            return self["w"]
        return max(0.0, np.random.normal(self["w"], self["sigma_w"]))


class NetworkBuilder:

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

    def __init__(self, mat_in, mat_out):
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

        assert(mat_in.shape[0] == mat_out.shape[0])
        self.N = mat_in.shape[0]
        self.m = mat_in.shape[1]
        self.n = mat_out.shape[1]

    def build_topology(self, k=-1, seed=None, topology_params={}):
        """
        Builds a network for a BiNAM that has been trained up to the k'th sample
        """

        # If k is smaller than zero, use the number of samples instead
        if k < 0 or k > self.N:
            k = self.N

        # Train the BiNAM from the last trained sample last_k up to k
        if self.mem == None or self.last_k > k:
            self.mem = binam.BiNAM(self.m, self.n)
        for l in xrange(self.last_k, k):
            self.mem.train(self.mat_in[l], self.mat_out[l])
        self.last_k = k

        # Build input and output neurons
        t = TopologyParameters(topology_params)
        s = t["multiplicity"]
        net = pynl.Network()
        for i in xrange(self.m):
            for j in xrange(s):
                net.add_source()
        for i in xrange(self.n):
            for j in xrange(s):
                net.add_neuron(params=t.draw(), _type=t["neuron_type"],
                        record=pynl.SIG_SPIKES)

        def in_coord(i, k=1):
            return (i * s + k, 0)

        def out_coord(j, k=1):
            return (self.m * s + j * s + k, 0)

        # Add all connections
        for i in xrange(self.m):
            for j in xrange(self.n):
                if self.mem[i, j] != 0:
                    net.add_connections([
                        (in_coord(i, k), out_coord(j, l), t.draw_weight(), 0.0)
                        for k in xrange(s) for l in xrange(s)])
        return net

    def build_input(self, k=-1, time_offs=0, index_offs = 0, topology_params={},
            input_params={}):
        """
        Builds the input spike trains for the network with the given input
        parameters. Returns a list with spike times for each neuron as first
        return value and a similar list containing the sample index for each
        spike time.
        """

        # If k is smaller than zero, use the number of samples instead
        if k < 0 or k > self.N:
            k = self.N

        # Make sure mat_in is a numpy array
        if isinstance(self.mat_in, binam.BinaryMatrix):
            X = self.mat_in.get()
        else:
            X = np.array(self.mat_in, dtype=np.uint8)

        # Turn the given parameters into an InputParameters instance
        t = TopologyParameters(topology_params)
        s = t["multiplicity"]

        p = InputParameters(input_params)
        b = p["burst_size"]

        # Calculate the maximum number of spikes, create two two-dimensional
        # matrix which contain the spike times and the sample indics
        max_num_spikes = np.max(np.sum(X, 0)) * b
        min_t = np.inf
        T = np.zeros((s * self.m, max_num_spikes))
        K = np.zeros((s * self.m, max_num_spikes), dtype=np.int32)
        N = np.zeros(self.m, dtype=np.uint32)
        for l in xrange(k):
            for i in xrange(self.m):
                if X[l, i] != 0:
                    for j in xrange(s):
                        train = p.build_spike_train(l * p["time_window"])
                        min_t = np.min([min_t, np.min(train)])
                        T[i * s + j, (N[i]):(N[i] + b)] = train
                        K[i * s + j, (N[i]):(N[i] + b)] = l + index_offs
                    N[i] = N[i] + b

        # Offset the first spike time to "time_window"
        T = T - min_t + time_offs

        # Extract the lists of lists from the matrices
        input_times = [[] for _ in xrange(s * self.m)]
        input_indices = [[] for _ in xrange(s * self.m)]
        for i in xrange(self.m):
            for j in xrange(s):
                x = i * s + j
                I = np.argsort(T[x, 0:N[i]])
                input_times[x] = T[x, I].tolist()
                input_indices[x] = K[x, I].tolist()

        return input_times, input_indices

    def inject_input(self, topology, times):
        """
        Injects the given spike times into the network.
        """
        for i in xrange(len(times)):
            topology["populations"][i]["params"]["spike_times"] = times[i]
        return topology

    def build(self, k=-1, time_offs=0, topology_params={}, input_params={}):
        """
        Builds a network with the given topology and input data that is ready
        to be handed of to PyNNLess.
        """
        topology = self.build_topology(k, topology_params=topology_params)
        input_times, input_indices = self.build_input(k, time_offs=time_offs,
                topology_params=topology_params, input_params=input_params)
        return NetworkInstance(
                self.inject_input(topology, input_times),
                input_times=input_times,
                input_indices=input_indices)

class NetworkInstance(dict):

    def __init__(self, data={}, populations=[], connections=[],
            input_times=[], input_indices=[]):
        utils.init_key(self, data, "populations", populations)
        utils.init_key(self, data, "connections", connections)
        utils.init_key(self, data, "input_times", input_times)
        utils.init_key(self, data, "input_indices", input_indices)

    def match(self, output):
        """
        Extracts the output spike times and matches them with the sample indices
        """

        # Flaten and sort the input times and input indices for efficient search
        total_input_count = reduce(lambda x, y: x + y,
                map(len, self["input_times"]))
        input_times = np.zeros(total_input_count)
        input_indices = np.zeros(total_input_count, dtype=np.int32)
        for i, t in enumerate(itertools.chain.from_iterable(
                self["input_times"])):
            input_times[i] = t
        for i, t in enumerate(itertools.chain.from_iterable(
                self["input_indices"])):
            input_indices[i] = t
        I = np.argsort(input_times)
        input_times = input_times[I]
        input_indices = input_indices[I]

        # Build the output times
        input_count = len(self["input_times"])
        output_count = len(output) - input_count
        output_times = [[] for _ in xrange(output_count)]
        for i in xrange(output_count):
            output_times[i] = output[i + input_count]["spikes"][0]

        # Build the output indices
        output_indices = [[] for _ in xrange(output_count)]
        for i in xrange(output_count):
            output_indices[i] = [0 for _ in xrange(len(output_times[i]))]
            for j in xrange(len(output_times[i])):
                t = output_times[i][j]
                idx = bisect.bisect_left(input_times, t)
                if idx > 0:
                    idx = idx - 1
                if idx < len(input_times):
                    output_indices[i][j] = input_indices[idx]

        return output_times, output_indices
