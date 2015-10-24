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
Contains the class which is responsible for reading the top-level experiment
descriptor and producing the multiplexed network pools.
"""

import json
import re
import random
import numpy as np
import utils

from network import (DataParameters, TopologyParameters, InputParameters,
        OutputParameters, NetworkBuilder, NetworkPool)

class ExperimentException(Exception):
    pass

class Experiment(dict):
    """
    Class representing an entrie collection of experiments and all network
    parameters. The Experiment class is responsible for generating a set of
    NetworkPool instances according to the experiment descriptors. These
    NetworkPool instances can then be executed independently.
    """

    # Used internally in the "validate_keys" method
    PARAMETER_PROTOTYPES = {
        "data": DataParameters(),
        "topology": TopologyParameters(),
        "input": InputParameters()
    }

    def __init__(self, data={}, data_params={}, topology_params={},
            input_params={}, output_params={}, experiments=[]):
        utils.init_key(self, data, "data", data_params)
        utils.init_key(self, data, "topology", topology_params)
        utils.init_key(self, data, "input", input_params)
        utils.init_key(self, data, "output", output_params)
        utils.init_key(self, data, "experiments", experiments)

        self["data"] = DataParameters(self["data"])
        self["topology"] = TopologyParameters(self["topology"])
        self["input"] = InputParameters(self["input"])
        self["output"] = OutputParameters(self["output"])

        if not isinstance(self["experiments"], list):
            self["experiments"] = [self["experiments"]]
        for i in xrange(len(self["experiments"])):
            self["experiments"][i] = ExperimentDescriptor(
                    self["experiments"][i])

    @staticmethod
    def read_from_file(filename):
        with open(filename) as stream:
            data = utils.parse_json_with_comments(stream)
        return Experiment(data)

    @classmethod
    def validate_keys(cls, keys):
        for key in keys:
            parts = key.split('.')
            if (len(parts) < 2 or len(parts) > 3 or (not parts[0] in
                    cls.PARAMETER_PROTOTYPES.keys())):
                raise ExperimentException("Invalid parameter key \"" + key
                    + "\": Must be of form "
                    + str(cls.PARAMETER_PROTOTYPES.keys()) + ".PARAM1[.PARAM2]")
            if (not parts[1] in cls.PARAMETER_PROTOTYPES[parts[0]]):
                raise ExperimentException("Invalid parameter key \"" + key
                    + "\": Unknown parameter \"" + parts[1] + "\", known "
                    + "parameters are "
                    + str(cls.PARAMETER_PROTOTYPES[parts[0]].keys()))
            if len(parts) == 3:
                if not isinstance(cls.PARAMETER_PROTOTYPES[parts[0]][parts[1]],
                        dict):
                    raise ExperimentException("Invalid parameter key \"" + key
                        + "\": Parameter \"" + parts[1]
                        + "\" is not a dictionary")
                if (key.startswith("topology.params.") or
                        key.startswith("topology.param_noise.")):
                    if not parts[2] in ["cm", "tau_refrac", "v_spike",
                            "v_reset", "v_rest", "tau_m", "i_offset", "a", "b",
                            "delta_T", "tau_w", "v_thresh", "e_rev_E",
                            "tau_syn_E", "e_rev_I", "tau_syn_I", "g_leak"]:
                        raise ExperimentException("Invalid parameter key \""
                                + key + "\": " + parts[2] + "\" is not a valid "
                                + "neuron parameter")

    def build_parameters(self, experiment):
        # Build the input and topology parameters for a single experiment
        input_params = []
        topology_params = []

        # Make sure the experiment descriptor contains only valid keys
        self.validate_keys(experiment.get_keys())

        # Build all input parameter combinations
        in_keys, in_vecs = experiment.build_combinatorial_sweep_vectors(
                experiment.get_input_sweeps())
        n, k = in_vecs.shape
        input_params = [InputParameters(self["input"])
                for _ in xrange(max(1, n))]
        if k > 0:
            for j in xrange(k):
                key = in_keys[j].split('.')[1]
                for i in xrange(n):
                    input_params[i][key] = in_vecs[i, j]

        # Build all topology and data parameter combinations
        top_keys, top_vecs = experiment.build_combinatorial_sweep_vectors(
                experiment.get_topology_sweeps())
        n, k = top_vecs.shape
        topology_params = [{
                    "topology": TopologyParameters(self["topology"]),
                    "data": DataParameters(self["data"])}
                for _ in xrange(max(1, n))]
        if k > 0:
            for j in xrange(k):
                key = top_keys[j].split('.')
                for i in xrange(n):
                    d = topology_params[i]
                    for l in xrange(len(key) - 1):
                        d = d[key[l]]
                    d[key[-1]] = top_vecs[i, j]

        return input_params, topology_params

    @staticmethod
    def _check_shared_parameters_equal(shared_parameters, ps1, ps2):
        for p in shared_parameters:
            if (p in ps1) and (p in ps2) and (ps1[p] != ps2[p]):
                return False
        return True

    def build(self, simulator_info, simulator="", seed=None):
        """
        Builds all NetworkPool instances required to conduct the specified
        experiments.

        :param simulator_info: Information about the used simulator as returned
        from PyNNLess.get_simulator_info() -- contains the maximum number of
        neurons and the supported software concurrency.
        :param seed: seed to be used to spawn the seeds for the data generation.
        """

        # Spawn more random seeds
        old_state = utils.initialize_seed(seed)
        try:
            data_seed = np.random.randint(1 << 30)
            build_seed = np.random.randint(1 << 30)
        finally:
            utils.finalize_seed(old_state)

        # Add a dummy experiment if there are no experiments specified
        if len(self["experiments"]) == 0:
            self["experiments"] = [ExperimentDescriptor(name="eval")]

        # Create all NetworkPool instances
        pools = []
        for i, experiment in enumerate(self["experiments"]):
            # Gather the input and topology parameters for this experiment
            input_params_list, topology_params_list = \
                    self.build_parameters(experiment)

            # Generate an experiment name
            # TODO: Make sure the name is unique
            if experiment["name"] == "":
                experiment["name"] = "experiment_" + str(i)

            # Generate new pools for this experiment
            min_pool = len(pools)
            pidx = simulator_info["concurrency"]

            # Assemble a name for this repetition
            pools = pools + [NetworkPool(name=experiment["name"] + "." + str(c))
                    for c in xrange(simulator_info["concurrency"])]

            # Metadata to store along with the networks
            meta_data = {
                "experiment_idx": i,
                "experiment_name": experiment["name"],
                "experiment_size": (experiment["repeat"] *
                        (len(input_params_list) * len(topology_params_list))),
                "keys": experiment.get_keys(),
                "output_params": self["output"],
                "simulator": simulator
            }

            # Repeat the experiment as many times as specified in the "repeat"
            # parameter
            local_build_seed = build_seed
            for j in xrange(experiment["repeat"]):
                # Create a random permutation of the topology parameters list
                perm = range(0, len(topology_params_list))
                random.shuffle(perm)
                for k in xrange(len(topology_params_list)):
                    # Create a build instance coupled with the topology
                    # parameters
                    topology_params = topology_params_list[perm[k]]
                    builder = NetworkBuilder(
                            data_params=topology_params["data"],
                            seed=data_seed)

                    # Build a network instance and add it to the network pool
                    net = builder.build(
                            topology_params=topology_params["topology"],
                            input_params=input_params_list,
                            meta_data=meta_data,
                            seed=local_build_seed)

                    # Search for a pool to which the network should be added.
                    # Use the pool with the fewest neurons which still has
                    # space for this experiment.
                    target_pool_idx = -1
                    for l in xrange(min_pool, len(pools)):
                        if ((target_pool_idx == -1 or pools[l].neuron_count()
                                < pools[target_pool_idx].neuron_count()) and 
                                 pools[l].neuron_count() + net.neuron_count()
                                    < simulator_info["max_neuron_count"]):
                            # If uniform parameter are required (Spikey), check
                            # whether the target network parameters are the same
                            # as the current network parameters
                            if pools[l].neuron_count() > 0:
                                if self._check_shared_parameters_equal(
                                        simulator_info["shared_parameters"],
                                        pools[l]["topology_params"][0]["params"],
                                        topology_params["topology"]["params"]):
                                    target_pool_idx = l
                            else:
                                target_pool_idx = l

                    # No free pool has been found, add a new one
                    if target_pool_idx == -1:
                        pool_name = experiment["name"] + "." + str(pidx)
                        pools.append(NetworkPool(name=pool_name))
                        pidx = pidx + 1
                        target_pool_idx = len(pools) - 1

                    # Add the network to the pool
                    pools[target_pool_idx].add_network(net)

                    # Advance the build_seed -- the input and topology
                    # parameters should still vary between trials,
                    # but reproducably
                    local_build_seed = local_build_seed * 2

        # Return non-empty pool instances
        return filter(lambda x: x.neuron_count() > 0, pools)

class ExperimentDescriptor(dict):

    def __init__(self, data={}, name="", repeat=1, sweeps={}):
        """
        Creates an experiment descriptor, which consists of a repeat count and
        a number of sweep descriptors for the dimensions that should be varied.
        """
        utils.init_key(self, data, "name", name)
        utils.init_key(self, data, "repeat", repeat)
        utils.init_key(self, data, "sweeps", sweeps)

    def get_sweeps(self, prefixes=[""]):
        sweeps = {}
        for key in self["sweeps"].keys():
            has_prefix = False
            for prefix in prefixes:
                has_prefix = has_prefix or key.startswith(prefix)
            if has_prefix:
                if (isinstance(self["sweeps"][key], dict)):
                    sweeps[key] = (ExperimentSweep(self["sweeps"][key])
                            .get_range())
                else:
                    sweeps[key] = np.array(self["sweeps"][key])
        return sweeps

    def get_keys(self):
        return self["sweeps"].keys()

    def get_input_sweeps(self):
        """
        Returns the input parameter sweeps.
        """
        return self.get_sweeps(["input."])

    def get_topology_sweeps(self):
        """
        Returns the topology/data parameter sweeps.
        """
        return self.get_sweeps(["data.", "topology."])

    @staticmethod
    def build_combinatorial_sweep_vectors(sweeps):
        """
        Given a dictionary containing the sweep, creates a matrix containing row
        vectors for every parameter combination.
        """
        keys = sweeps.keys()

        # Make sure we have no zero-length sweeps
        for i, key in enumerate(keys):
            if (len(sweeps[key]) == 0):
                del keys[i]

        ndims = len(keys)
        dims = [0] * ndims
        dim_lens = [len(sweeps[key]) for key in keys]
        count = 0 if ndims == 0 else reduce(lambda x, y: x * y, dim_lens)
        vecs = np.zeros((count, ndims))
        for i in xrange(count):
            # Copy the current values described by the index vector dims into
            # the result matrix
            for j in xrange(ndims):
                vecs[i, j] = sweeps[keys[j]][dims[j]]

            # Increment the index vector with overflow between dimensions
            for j in xrange(ndims):
                dims[j] = dims[j] + 1
                if (dims[j] == dim_lens[j]):
                    dims[j] = 0
                else:
                    break

        return keys, vecs

class ExperimentSweep(dict):
    """
    Represents a parameter range with minimum and maximum value and the number
    of steps between those two values.
    """

    def __init__(self, data={}, vmin=0, vmax=1, count=10):
        utils.init_key(self, data, "min", vmin)
        utils.init_key(self, data, "max", vmax)
        utils.init_key(self, data, "count", count)

    def get_range(self):
        return np.linspace(self["min"], self["max"], self["count"])

