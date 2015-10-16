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
import entropy
import data
import bisect
import itertools
import numpy as np
import pynnless as pynl
import pynnless.pynnless_utils as utils

def initialize_seed(seed, seq=1):
    """
    Initializes the numpy random number generator seed with the given seed
    value. The seed value may be None, in which case no changes the the numpy
    seed are made. Returns the old random generator state or None if no change
    was made.
    """
    if seed is None:
        return None
    old_state = np.random.get_state()
    np.random.seed(seed * (seq + 1))
    return old_state

def finalize_seed(old_state):
    """
    Restores the numpy random seed to its old value, or does nothin if the given
    value is "None".
    """
    if (old_state != None):
        np.random.set_state(old_state)

class DataParameters(dict):
    """
    Stores the data meta-parameters (number of input and output bits,
    number of samples, number of set ones)
    """

    def __init__(self, data={}, n_bits_in=16, n_bits_out=16, n_ones_in=3,
            n_ones_out=3, n_samples = 100):
        """
        Fills the network structure with the given parameters.

        :param n_bits_in: number of input bits.
        :param n_bits_out: number of output bits.
        :param n_ones_in: number of ones in the input per sample.
        :param n_ones_out: number of ones in the output per sample.
        :param n_samples: number of samples.
        """
        utils.init_key(self, data, "n_bits_in", n_bits_in)
        utils.init_key(self, data, "n_bits_out", n_bits_out)
        utils.init_key(self, data, "n_ones_in", n_ones_in)
        utils.init_key(self, data, "n_ones_out", n_ones_out)
        utils.init_key(self, data, "n_samples", n_samples)

class InputParameters(dict):
    """
    Contains parameters which concern the generation of input data.
    """

    def __init__(self, data={}, burst_size=1, time_window=100.0, isi=1.0,
            sigma_t=0.0, sigma_t_offs=0.0, p0=0.0, p1=0.0):
        """
        Constructor of the InputParameters class. If a dictionary or another
        InputParameters class is passed as first parameter, elements from this
        instance are copied with a precedence over the specified parameters.

        :param data: dictionary from which parameters are copied preferably.
        :param burst_size: number of spikes representing a "one".
        :param time_window: time between the presentation of two input samples.
        :param isi: inter spike interval between spike of a burst.
        :param sigma_t: spike jitter
        :param sigma_t_offs: spike offset jitter
        :param p0: probability with which individual spikes are omitted.
        :param p1: probability with which spikes from a "one" are randomly
        introduced for a sample containing zeros.
        """
        utils.init_key(self, data, "burst_size", burst_size)
        utils.init_key(self, data, "time_window", time_window)
        utils.init_key(self, data, "isi", isi)
        utils.init_key(self, data, "sigma_t", sigma_t)
        utils.init_key(self, data, "sigma_t_offs", sigma_t_offs)
        utils.init_key(self, data, "p0", p0)
        utils.init_key(self, data, "p1", p1)

    def build_spike_train(self, value=1, offs=0.0):
        """
        Builds a spike train representing the given binary value according to
        the parameters stored in the InputParameters dictionary.

        :param value: the binary value that should be represented by this spike
        train. If zero, spikes are only generated according to the probability
        p1.
        :param offs: total time offset.
        """
        res = []

        # Draw the actual spike offset
        if (self["sigma_t_offs"] > 0):
            offs = np.random.normal(offs, self["sigma_t_offs"])

        # Calculate the time of each spike
        if value == 1:
            p = self["p0"]
        else:
            p = 1.0 - self["p1"]

        for i in xrange(self["burst_size"]):
            if (np.random.uniform() >= p):
                jitter = 0
                if (self["sigma_t"] > 0):
                    jitter = np.random.normal(0, self["sigma_t"])
                res.append(offs + i * self["isi"] + jitter)
        res.sort()
        return res

class OutputParameters(dict):
    """
    Contains parameters describing the expected output.
    """

    def __init__(self, data={}, burst_size=1):
        """
        Constructor of the OutputParameters class. Either copies the parameters
        from the given "data" dictionary or uses the given arguments.

        :param data: dictionary the arguments are preferably copied from.
        :param burst_size: number of expected output spikes in a neuron.
        """
        utils.init_key(self, data, "burst_size", burst_size)

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

    # Input data matrix
    mat_in = None

    # Output data matrix
    mat_out = None

    # Data parameters
    data_params = None

    # Internally cached BiNAM instance
    mem = None

    # Last sample until which the BiNAM has been trained
    last_k = 0

    @staticmethod
    def _n_ones(mat):
        m, n = mat.shape
        if m > 0:
            # Use the number of ones in the first row
            return np.sum(mat[0])
        else:
            return 0

    def __init__(self, mat_in=None, mat_out=None, data_params=None, seed=None):
        """
        Constructor of the NetworkBuilder class -- the NetworkBuilder collects
        information about a network (storage matrix, noise parameters and input
        data).

        :param mat_in: Nxm matrix containing the input data
        :param mat_out: Nxn matrix containing the output data
        """

        # Make sure that either data parameters are given or an input
        # and output matrix
        assert((mat_in is None) == (mat_out is None))
        assert((mat_in is None) != (data_params is None))

        if mat_in is None:
            # Use the supplied data parameters -- generate the data matrices
            self.data_params = DataParameters(data_params)

            # Set the random number generator seed and generate the data
            old_seed = initialize_seed(seed, 1)
            try:
                self.mat_in = data.generate(
                    n_bits = self.data_params["n_bits_in"],
                    n_ones = self.data_params["n_ones_in"],
                    n_samples = self.data_params["n_samples"])
            finally:
                finalize_seed(old_seed)

            # Reset the random number generator seed to make sure that the first
            # n_samples are always the same
            old_seed = initialize_seed(seed, 2)
            try:
                self.mat_out = data.generate(
                    n_bits = self.data_params["n_bits_out"],
                    n_ones = self.data_params["n_ones_out"],
                    n_samples = self.data_params["n_samples"])
            finally:
                finalize_seed(old_seed)

        else:
            # If a matrices are given, derive the data parameters from those
            self.mat_in = mat_in
            self.mat_out = mat_out

            assert(mat_in.shape[0] == mat_out.shape[0])
            self.data_params = DataParameters(
                n_bits_in = mat_in.shape[1],
                n_bits_out = mat_out.shape[1],
                n_ones_in = self._n_ones(mat_in),
                n_ones_out = self._n_ones(mat_out),
                n_samples = mat_in.shape[0])

    def build_topology(self, k=-1, seed=None, topology_params={}):
        """
        Builds a network for a BiNAM that has been trained up to the k'th sample
        """

        # Fetch the data parameters for convenient access
        N = self.data_params["n_samples"]
        m = self.data_params["n_bits_in"]
        n = self.data_params["n_bits_out"]

        # If k is smaller than zero, use the number of samples instead
        if k < 0 or k > N:
            k = N

        # Train the BiNAM from the last trained sample last_k up to k
        if self.mem == None or self.last_k > k:
            self.mem = binam.BiNAM(m, n)
        for l in xrange(self.last_k, k):
            self.mem.train(self.mat_in[l], self.mat_out[l])
        self.last_k = k

        # Build input and output neurons
        t = TopologyParameters(topology_params)
        s = t["multiplicity"]
        net = pynl.Network()
        for i in xrange(m):
            for j in xrange(s):
                net.add_source()
        for i in xrange(n):
            for j in xrange(s):
                net.add_neuron(params=t.draw(), _type=t["neuron_type"],
                        record=pynl.SIG_SPIKES)

        def in_coord(i, k=1):
            return (i * s + k, 0)

        def out_coord(j, k=1):
            return (m * s + j * s + k, 0)

        # Add all connections
        for i in xrange(m):
            for j in xrange(n):
                if self.mem[i, j] != 0:
                    net.add_connections([
                        (in_coord(i, k), out_coord(j, l), t.draw_weight(), 0.0)
                        for k in xrange(s) for l in xrange(s)])
        return net

    @staticmethod
    def build_spike_trains(mat, k=-1, time_offs=0, topology_params={},
            input_params={}, input_params_delay=10):
        """
        Builds a list of spike trains as encoded in the given matrix, consisting
        of one sample per row.
        """

        def rmin(xs):
            """
            Roboust min function, returns Inf if xs is empty.
            """
            return np.inf if len(xs) == 0 else min(xs)

        # Fetch the data parameters for convenient access
        N = mat.shape[0]
        m = mat.shape[1]

        # If k is smaller than zero, use the number of samples instead
        if k < 0 or k > N:
            k = N

        # Make sure mat is a numpy array
        if isinstance(mat, binam.BinaryMatrix):
            X = mat.get()
        else:
            X = np.array(mat, dtype=np.uint8)

        # Make sure input_params is a list
        if not isinstance(input_params, list):
            input_params = [input_params]

        # Fetch the multiplicity s from the topology parameters
        s = TopologyParameters(topology_params)["multiplicity"]

        # Resulting times and indices
        input_times = [[] for _ in xrange(s * m)]
        input_indices = [[] for _ in xrange(s * m)]

        # Handle all input parameter sets
        t = 0
        sIdx = 0
        min_t = np.inf
        for ip in input_params:
            # Convert the parameters into a normalized InputParameters instance
            p = InputParameters(ip)

            # Calculate the maximum number of spikes, create two two-dimensional
            # matrix which contain the spike times and the sample indics
            for l in xrange(k):
                for i in xrange(m):
                    for j in xrange(s):
                        train = p.build_spike_train(value=X[l, i], offs=t)
                        idx = i * s + j
                        input_times[idx] = input_times[idx] + train
                        input_indices[idx] = (input_indices[idx] +
                                [sIdx] * len(train))
                sIdx = sIdx + 1
                t = t + p["time_window"]
            t = t + p["time_window"] * input_params_delay

        # Offset the first spike time to time_offs
        min_t = min(map(rmin, input_times))
        input_times = map(lambda ts: map(lambda t: t - min_t + time_offs, ts),
                input_times)

        # Store the sample indices at which new input parameter sets start
        input_split = range(k, k * (len(input_params) + 1), k)

        return input_times, input_indices, input_split

    def build_input(self, k=-1, time_offs=0, topology_params={},
            input_params={}, input_params_delay=10):
        """
        Builds the input spike trains for the network with the given input
        parameter sets. Returns a list with spike times for each neuron as first
        return value and a similar list containing the sample index for each
        spike time. Note that input_params may be an array of parameter sets --
        in this case multiple input spike trains are created.
        """
        return self.build_spike_trains(self.mat_in, k, time_offs,
                topology_params, input_params, input_params_delay)

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

        # If k is smaller than zero, use the number of samples instead
        if k < 0 or k > N:
            k = self.data_params["n_samples"]

        topology = self.build_topology(k, topology_params=topology_params)
        input_times, input_indices, input_split = self.build_input(k,
                time_offs=time_offs,
                topology_params=topology_params,
                input_params=input_params)
        return NetworkInstance(
                self.inject_input(topology, input_times),
                input_times = input_times,
                input_indices = input_indices,
                input_split = input_split,
                data_params = DataParameters(self.data_params, n_samples=k),
                topology_params = topology_params)

class NetworkInstance(dict):
    """
    Concrete instance of a BiNAM network that can be passed to the PyNNLess run
    method. A NetworkInstance can contain a time multiplex of simulations
    (with different input parameters). It provides a build_analysis method which
    splits the time multiplexed results into individual NetworkAnalysis objects.
    """

    def __init__(self, data={}, populations=[], connections=[],
            input_times=[], input_indices=[], input_split=[], data_params={},
            topology_params={}):
        utils.init_key(self, data, "populations", populations)
        utils.init_key(self, data, "connections", connections)
        utils.init_key(self, data, "input_times", input_times)
        utils.init_key(self, data, "input_indices", input_indices)
        utils.init_key(self, data, "input_split", input_split)
        utils.init_key(self, data, "data_params", data_params)
        utils.init_key(self, data, "topology_params", topology_params)

        self["data_params"] = DataParameters(self["data_params"])
        self["topology_params"] = TopologyParameters(self["topology_params"])

    @staticmethod
    def flaten(times, indices, sort_by_sample=False):
        """
        Flatens a list of spike times and corresponding indices to three
        one-dimensional arrays containing the spike time, sample indices and
        neuron indices.
        """

        # Calculate the total count of spikes
        count = reduce(lambda x, y: x + y, map(len, times))

        # Initialize the flat arrays containing the times, indices and neuron
        # indices
        tF = np.zeros(count)
        kF = np.zeros(count, dtype=np.int32)
        nF = np.zeros(count, dtype=np.int32)

        # Iterate over all neurons and all spike times
        c = 0
        for i in xrange(len(times)):
            for j in xrange(len(times[i])):
                tF[c] = times[i][j]
                kF[c] = indices[i][j]
                nF[c] = i
                c = c + 1

        # Sort the arrays by spike time or sample
        if sort_by_sample:
            I = np.lexsort((kF, tF))
        else:
            I = np.argsort(tF)
        tF = tF[I]
        kF = kF[I]
        nF = nF[I]
        return tF, kF, nF

    @staticmethod
    def match_static(input_times, input_indices, output):
        """
        Extracts the output spike times from the simulation output and
        calculates the sample index for each output spike.
        """

        # Flaten and sort the input times and input indices for efficient search
        tIn, kIn, _ = NetworkInstance.flaten(input_times, input_indices)

        # Build the output times
        input_count = len(input_times)
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
                idx = bisect.bisect_left(tIn, t)
                if idx > 0:
                    idx = idx - 1
                if idx < len(tIn):
                    output_indices[i][j] = kIn[idx]

        return output_times, output_indices

    def match(self, output):
        """
        Extracts the output spike times from the simulation output and
        calculates the sample index for each output spike.
        """

        return self.match_static(self["input_times"], self["input_indices"],
                output)

    @staticmethod
    def split(times, indices, k0, k1):
        times_part = [[] for _ in xrange(len(times))]
        indices_part = [[] for _ in xrange(len(indices))]
        for i in xrange(len(times)):
            for j in xrange(len(times[i])):
                if indices[i][j] >= k0 and indices[i][j] < k1:
                    times_part[i].append(times[i][j])
                    indices_part[i].append(indices[i][j] - k0)
        return times_part, indices_part

    @staticmethod
    def build_analysis_static(input_times, input_indices, output, data_params,
            topology_params, input_split=[]):
        # Fetch the output times and output indices
        output_times, output_indices = NetworkInstance.match_static(input_times,
                input_indices, output)

        # Only assume a single split if no input_split descriptor is given
        if (len(input_split) == 0):
            input_split = [max(map(max, input_indices)) + 1]

        # Split input and output spikes according to the input_split map, create
        # a NetworkAnalysis instance for each split
        res = []
        k0 = 0
        for k in input_split:
            input_times_part, input_indices_part = NetworkInstance.split(
                    input_times, input_indices, k0, k)
            output_times_part, output_indices_part = NetworkInstance.split(
                    output_times, output_indices, k0, k)
            res.append(NetworkAnalysis(
                    input_times = input_times_part,
                    input_indices = input_indices_part,
                    output_times = output_times_part,
                    output_indices = output_indices_part,
                    data_params = data_params,
                    topology_params = topology_params))
            k0 = k
        return res

    def build_analysis(self, output):
        return self.build_analysis_static(self["input_times"],
                self["input_indices"], output, self["data_params"],
                self["topology_params"], self["input_split"])

class NetworkPool(dict):
    """
    The NetworkPool represents a spatial multiplex of multiple networks. It
    allows to add an arbitrary count of NetworkInstance objects and provides
    a "build_analysis" method which splits the network output into individual
    NetworkAnalysis object for each time/spatial multiplex.
    """

    def __init__(self, data={}, populations=[], connections=[],
            input_times=[], input_indices=[], input_split=[], spatial_split=[],
            data_params=[], topology_params=[]):
        utils.init_key(self, data, "populations", populations)
        utils.init_key(self, data, "connections", connections)
        utils.init_key(self, data, "input_times", input_times)
        utils.init_key(self, data, "input_indices", input_indices)
        utils.init_key(self, data, "input_split", input_split)
        utils.init_key(self, data, "spatial_split", spatial_split)
        utils.init_key(self, data, "data_params", data_params)
        utils.init_key(self, data, "topology_params", topology_params)

        # Fix things up in case a NetworkInstance was passed to the constructor
        if (len(self["spatial_split"]) == 0 and len(self["populations"]) > 0):
            self["input_split"] = [self["input_split"]]
            self["data_params"] = [self["data_params"]]
            self["topology_params"] = [self["topology_params"]]
            self["spatial_split"].append({
                    "population": len(self["populations"]),
                    "input": len(self["input_times"])
                });

    def add_network(self, network):
        """
        Adds a new NetworkInstance to the execution pool.
        """

        # Old population and connection count
        nP0 = len(self["populations"])
        nC0 = len(self["connections"])

        # Append the network to the pool network
        self["populations"] = self["populations"] + network["populations"]
        self["connections"] = self["connections"] + network["connections"]
        self["input_times"] = self["input_times"] + network["input_times"]
        self["input_indices"] = self["input_indices"] + network["input_indices"]
        self["input_split"].append(network["input_split"])
        self["data_params"].append(network["data_params"])
        self["topology_params"].append(network["topology_params"])

        # Add a "spatial_split" -- this allows to dissect the network into its
        # original parts after the result is available
        nP1 = len(self["populations"])
        nC1 = len(self["connections"])
        nI1 = len(self["input_times"])
        self["spatial_split"].append({"population": nP1, "input": nI1});

        # Adapt the connection population indices of the newly added connections
        for i in xrange(nC0, nC1):
            c = self["connections"][i]
            self["connections"][i] = ((c[0][0] + nP0, c[0][1]),
                (c[1][0] + nP0, c[1][1]), c[2], c[3])

    def add_networks(self, networks):
        """
        Adds a list of NetworkInstance instances to the execution pool.
        """
        for network in networks:
            self.add_network(network)

    def build_analysis(self, output):
        """
        Performs spatial and temporal demultiplexing of the conducted
        experiments.
        """

        # Iterate over each spatial split and gather the analysis instances
        res = []
        last_split = {"population": 0, "input": 0}
        for i, split in enumerate(self["spatial_split"]):
            # Split the input times and the input indices at the positions
            # stored in the split descriptor
            input_times = self["input_times"][
                    last_split["input"]:split["input"]]
            input_indices = self["input_indices"][
                    last_split["input"]:split["input"]]

            # Split the output for the stored population range
            output_part = output[last_split["population"]:split["population"]]

            # Find the input_split descriptor, use an empty descriptor if no
            # valid input_split descriptor was stored.
            input_split = []
            if (i < len(self["input_split"]) and
                    isinstance(self["input_split"][i], list)):
                input_split = self["input_split"][i]

            # Fetch the i-th "data_params" and "topology_params" instance
            data_params = self["data_params"][i]
            topology_params = self["topology_params"][i]

            # Let the NetworkInstance class build the analysis instances. This
            # class is responsible for performing the temporal demultiplexing.
            res = res + NetworkInstance.build_analysis_static(
                input_times = input_times,
                input_indices = input_indices,
                output = output_part,
                data_params = data_params,
                topology_params = topology_params,
                input_split = input_split)
            last_split = split
        return res

class NetworkAnalysis(dict):
    """
    Contains the input and output spikes gathered for a single test run.
    Provides methods for performing a storage capacity and latency analysis.
    """

    def __init__(self, data={}, input_times=[], input_indices=[],
            output_times=[], output_indices=[], data_params={},
            topology_params={}):
        utils.init_key(self, data, "input_times", input_times)
        utils.init_key(self, data, "input_indices", input_indices)
        utils.init_key(self, data, "output_times", output_times)
        utils.init_key(self, data, "output_indices", output_indices)
        utils.init_key(self, data, "data_params", data_params)
        utils.init_key(self, data, "topology_params", topology_params)

        self["data_params"] = DataParameters(self["data_params"])
        self["topology_params"] = TopologyParameters(self["topology_params"])

    def calculate_latencies(self):
        """
        Calculates the latency of each sample for both an input and output spike
        is available. Returns a list of latency values with an entry for each
        sample. The latency for samples without a response is set to infinity.
        """

        # Flaten the input and output times and indices
        tIn, kIn, _ = NetworkInstance.flaten(self["input_times"],
                self["input_indices"], sort_by_sample=True)
        tOut, kOut, _ = NetworkInstance.flaten(self["output_times"],
                self["output_indices"], sort_by_sample=True)

        # Fetch the number of samples
        N = self["data_params"]["n_samples"]
        res = np.zeros(N)

        # Calculate the latency for each sample
        for k in xrange(N):
            # Fetch index of the latest input and output spike time for sample k
            iInK = bisect.bisect_right(kIn, k) - 1
            iOutK = bisect.bisect_right(kOut, k) - 1

            # Make sure the returned values are valid and actually refer to the
            # current sample
            if (iInK < 0 or iOutK < 0 or kIn[iInK] != k or kOut[iOutK] != k):
                res[k] = np.inf
            else:
                res[k] = tOut[iOutK] - tIn[iInK]
        return res

    def calculate_output_matrix(self, output_params={}):
        """
        Calculates a matrix containing the actually calculated output samples.
        """

        # Flaten the output spike sample indices and neuron indices
        _, kOut, nOut = NetworkInstance.flaten(self["output_times"],
                self["output_indices"], sort_by_sample=True)

        # Fetch the neuron multiplicity
        s = self["topology_params"]["multiplicity"]

        # Create the output matrix
        N = self["data_params"]["n_samples"]
        n = self["data_params"]["n_bits_out"]
        res = np.zeros((N, n))

        # Iterate over each sample
        for k in xrange(N):
            # Fetch the indices in the flat array corresponding to that sample
            i0 = bisect.bisect_left(kOut, k)
            i1 = bisect.bisect_right(kOut, k)

            # For each output spike increment the corresponding entry in the
            # output matrix
            for i in xrange(i0, i1):
                res[k, nOut[i] // s] = res[k, nOut[i] // s] + 1

        # Scale the result matrix according to the output_burst_size
        return res / float(s * OutputParameters(output_params)["burst_size"])

    def calculate_storage_capactiy(self, mat_out_expected, output_params={}):
        """
        Calculates the storage capacity of the BiNAM, given the expected output
        data and number of ones in the output. Returns the information, the
        output matrix and the error counts.
        """
        mat_out = self.calculate_output_matrix(output_params)
        N, n = mat_out.shape
        errs = entropy.calculate_errs(mat_out, mat_out_expected)
        I = entropy.entropy_hetero(errs, n, self["data_params"]["n_ones_out"])
        return I, mat_out, errs

