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

from pynam.binam import BinaryMatrix
from pynam.binam_network import (
        InputParameters,
        TopologyParameters,
        NetworkBuilder,
        NetworkInstance,
        NetworkPool,
        NetworkAnalysis)

#
# Test utility functions
#

def test_data():
    mat_in = BinaryMatrix(3, 5)
    mat_in[0, 4] = 1
    mat_in[0, 2] = 1
    mat_in[1, 0] = 1
    mat_in[1, 1] = 1
    mat_in[2, 3] = 1
    mat_in[2, 2] = 1

    mat_out = BinaryMatrix(3, 5)
    mat_out[0, 2] = 1
    mat_out[0, 3] = 1
    mat_out[1, 1] = 1
    mat_out[1, 4] = 1
    mat_out[2, 0] = 1
    mat_out[2, 1] = 1

    return mat_in, mat_out

def test_data2():
    mat_in = BinaryMatrix(2, 4)
    mat_in[0, 3] = 1
    mat_in[0, 1] = 1
    mat_in[1, 0] = 1
    mat_in[1, 1] = 1

    mat_out = BinaryMatrix(2, 6)
    mat_out[0, 0] = 1
    mat_out[0, 1] = 1
    mat_out[0, 4] = 1
    mat_out[1, 2] = 1
    mat_out[1, 3] = 1
    mat_out[1, 5] = 1

    return mat_in, mat_out

time_mux_output_data = [
            {}, {}, {}, {}, {},
            {"spikes": [[0.0, 100.0, 1301.0, 1400.0]]},
            {"spikes": [[101.0, 1401.0]]},
            {"spikes": [[300.0, 301.0, 1600.0, 1601.0]]},
            {"spikes": [[200.0, 1500.0]]},
            {"spikes": [[201.0, 101.0, 1501.0, 1401.0]]}
        ]

def test_time_mux_res(self, analysis_instances):
    self.assertEqual(2, len(analysis_instances))

    self.assertEqual(
        analysis_instances[0]["input_times"],
        [[100.0], [100.0], [0.0, 200.0], [200.0], [0.0]])
    self.assertEqual(
        analysis_instances[0]["input_indices"],
        [[1], [1], [0, 2], [2], [0]])
    self.assertEqual(
        analysis_instances[0]["output_times"],
        [[0.0, 100.0], [101.0], [300.0, 301.0], [200.0], [201.0, 101.0]])
    self.assertEqual(
        analysis_instances[0]["output_indices"],
        [[0, 0], [1], [2, 2], [1], [2, 1]])

    self.assertEqual(
        analysis_instances[1]["input_times"],
        [[1400.0], [1400.0], [1300.0, 1500.0], [1500.0], [1300.0]])
    self.assertEqual(
        analysis_instances[1]["input_indices"],
        [[1], [1], [0, 2], [2], [0]])
    self.assertEqual(
        analysis_instances[1]["output_times"],
        [[1301.0, 1400.0], [1401.0], [1600.0, 1601.0], [1500.0],
         [1501.0, 1401.0]])
    self.assertEqual(
        analysis_instances[1]["output_indices"],
        [[0, 0], [1], [2, 2], [1], [2, 1]])

def stub_simulation(mat_out, data_params={}, input_params={},
        topology_params={}, latency=1.0):
    # Build the output spike trains
    trains, _, _ = NetworkBuilder.build_spike_trains(mat_out, time_offs=latency,
            input_params=input_params, topology_params=topology_params)

    # Encapsulate the perfect spike trains in the PyNNLess output format, add
    # empty elements for the input populations
    nin = (data_params["n_bits_in"]
            * TopologyParameters(topology_params)["multiplicity"])
    return ([{} for _ in xrange(nin)] +
            [{"spikes": [train]} for train in trains])

#
# Unit tests
#

class TestInputParameters(unittest.TestCase):

    def test_build_spike_train(self):
        params = InputParameters()
        res = params.build_spike_train()
        numpy.testing.assert_equal([0.0], res)

        params = InputParameters(burst_size=2)
        res = params.build_spike_train()
        numpy.testing.assert_equal([0.0, 1.0], res)

        params = InputParameters(burst_size=3, isi=4)
        res = params.build_spike_train()
        numpy.testing.assert_equal([0.0, 4.0, 8.0], res)

class TestNetworkBuilder(unittest.TestCase):

    def test_init_mat(self):
        mat_in, mat_out = test_data2()
        net = NetworkBuilder(mat_in, mat_out)

        self.assertEqual(4, net.data_params["n_bits_in"])
        self.assertEqual(6, net.data_params["n_bits_out"])
        self.assertEqual(2, net.data_params["n_ones_in"])
        self.assertEqual(3, net.data_params["n_ones_out"])
        self.assertEqual(2, net.data_params["n_samples"])

    def test_init_data(self):
        net = NetworkBuilder(data_params={
            "n_bits_in": 8,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 5,
            "n_samples": 20
        })

        net2 = NetworkBuilder(net.mat_in, net.mat_out)

        self.assertEqual(8, net2.data_params["n_bits_in"])
        self.assertEqual(10, net2.data_params["n_bits_out"])
        self.assertEqual(3, net2.data_params["n_ones_in"])
        self.assertEqual(5, net2.data_params["n_ones_out"])
        self.assertEqual(20, net2.data_params["n_samples"])

    def test_init_data_seed(self):
        net1 = NetworkBuilder(data_params={
            "n_bits_in": 8,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 5,
            "n_samples": 20
        }, seed = 15412)
        net2 = NetworkBuilder(data_params={
            "n_bits_in": 8,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 5,
            "n_samples": 40
        }, seed = 15412)

        # The first twenty samples must be equal
        np.testing.assert_equal(net1.mat_in, net2.mat_in[0:20])
        np.testing.assert_equal(net1.mat_out, net2.mat_out[0:20])


    def test_build_topology(self):
        mat_in, mat_out = test_data()
        net = NetworkBuilder(mat_in, mat_out)
        topo = net.build_topology(topology_params={"params": {"cm": 0.2}})
        self.assertEqual(set(topo["connections"]), set([
            ((0, 0), (6, 0), 0.03, 0.0),
            ((0, 0), (9, 0), 0.03, 0.0),
            ((1, 0), (6, 0), 0.03, 0.0),
            ((1, 0), (9, 0), 0.03, 0.0),
            ((2, 0), (5, 0), 0.03, 0.0),
            ((2, 0), (6, 0), 0.03, 0.0),
            ((2, 0), (7, 0), 0.03, 0.0),
            ((2, 0), (8, 0), 0.03, 0.0),
            ((3, 0), (5, 0), 0.03, 0.0),
            ((3, 0), (6, 0), 0.03, 0.0),
            ((4, 0), (7, 0), 0.03, 0.0),
            ((4, 0), (8, 0), 0.03, 0.0)]))
        self.assertEqual(topo["populations"], [
                {'count': 1,
                 'params': {'spike_times': []},
                 'type': 'SpikeSourceArray',
                 'record': []}
            ] * 5 + [
                {'count': 1,
                 'params': {
                    'tau_refrac': 0.1,
                    'tau_m': 20.0,
                    'e_rev_E': 0.0,
                    'i_offset': 0.0,
                    'cm': 0.2,
                    'e_rev_I': -70.0,
                    'v_thresh': -50.0,
                    'tau_syn_E': 5.0,
                    'v_rest': -65.0,
                    'tau_syn_I': 5.0,
                    'v_reset': -65.0
                },
                'record': ['spikes'],
                'type': 'IF_cond_exp'
                }
            ] * 5)

        net = NetworkBuilder(mat_in, mat_out)
        topo = net.build_topology(topology_params={"multiplicity": 2, "w": 0.1})
        self.assertEqual(set(topo["connections"]), set([
            ((0, 0), (12, 0), 0.1, 0.0),
            ((0, 0), (13, 0), 0.1, 0.0),
            ((1, 0), (12, 0), 0.1, 0.0),
            ((1, 0), (13, 0), 0.1, 0.0),
            ((0, 0), (18, 0), 0.1, 0.0),
            ((0, 0), (19, 0), 0.1, 0.0),
            ((1, 0), (18, 0), 0.1, 0.0),
            ((1, 0), (19, 0), 0.1, 0.0),
            ((2, 0), (12, 0), 0.1, 0.0),
            ((2, 0), (13, 0), 0.1, 0.0),
            ((3, 0), (12, 0), 0.1, 0.0),
            ((3, 0), (13, 0), 0.1, 0.0),
            ((2, 0), (18, 0), 0.1, 0.0),
            ((2, 0), (19, 0), 0.1, 0.0),
            ((3, 0), (18, 0), 0.1, 0.0),
            ((3, 0), (19, 0), 0.1, 0.0),
            ((4, 0), (10, 0), 0.1, 0.0),
            ((4, 0), (11, 0), 0.1, 0.0),
            ((5, 0), (10, 0), 0.1, 0.0),
            ((5, 0), (11, 0), 0.1, 0.0),
            ((4, 0), (12, 0), 0.1, 0.0),
            ((4, 0), (13, 0), 0.1, 0.0),
            ((5, 0), (12, 0), 0.1, 0.0),
            ((5, 0), (13, 0), 0.1, 0.0),
            ((4, 0), (14, 0), 0.1, 0.0),
            ((4, 0), (15, 0), 0.1, 0.0),
            ((5, 0), (14, 0), 0.1, 0.0),
            ((5, 0), (15, 0), 0.1, 0.0),
            ((4, 0), (16, 0), 0.1, 0.0),
            ((4, 0), (17, 0), 0.1, 0.0),
            ((5, 0), (16, 0), 0.1, 0.0),
            ((5, 0), (17, 0), 0.1, 0.0),
            ((6, 0), (10, 0), 0.1, 0.0),
            ((6, 0), (11, 0), 0.1, 0.0),
            ((7, 0), (10, 0), 0.1, 0.0),
            ((7, 0), (11, 0), 0.1, 0.0),
            ((6, 0), (12, 0), 0.1, 0.0),
            ((6, 0), (13, 0), 0.1, 0.0),
            ((7, 0), (12, 0), 0.1, 0.0),
            ((7, 0), (13, 0), 0.1, 0.0),
            ((8, 0), (14, 0), 0.1, 0.0),
            ((8, 0), (15, 0), 0.1, 0.0),
            ((9, 0), (14, 0), 0.1, 0.0),
            ((9, 0), (15, 0), 0.1, 0.0),
            ((8, 0), (16, 0), 0.1, 0.0),
            ((8, 0), (17, 0), 0.1, 0.0),
            ((9, 0), (16, 0), 0.1, 0.0),
            ((9, 0), (17, 0), 0.1, 0.0)]))

    def test_build_input(self):
        mat_in, mat_out = test_data()
        net = NetworkBuilder(mat_in, mat_out)
        times, indices, split = net.build_input()
        self.assertEqual([[100.0], [100.0], [0.0, 200.0], [200.0], [0.0]], times)
        self.assertEqual([[1], [1], [0, 2], [2], [0]], indices)
        self.assertEqual([3], split)

        times, indices, split = net.build_input(
                topology_params={"multiplicity": 2})
        self.assertEqual([[100.0], [100.0], [100.0], [100.0], [0.0, 200.0],
                [0.0, 200.0], [200.0], [200.0], [0.0], [0.0]], times)
        self.assertEqual([[1], [1], [1], [1], [0, 2], [0, 2],
                [2], [2], [0], [0]], indices)
        self.assertEqual([3], split)

        times, indices, split = net.build_input(
                topology_params={"multiplicity": 2},
                input_params={"burst_size": 3})
        self.assertEqual([[100.0, 101.0, 102.0], [100.0, 101.0, 102.0],
                [100.0, 101.0, 102.0], [100.0, 101.0, 102.0],
                [0.0, 1.0, 2.0, 200.0, 201.0, 202.0],
                [0.0, 1.0, 2.0, 200.0, 201.0, 202.0],
                [200.0, 201.0, 202.0], [200.0, 201.0, 202.0],
                [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], times)
        self.assertEqual([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                [0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 2], [2, 2, 2], [2, 2, 2],
                [0, 0, 0], [0, 0, 0]], indices)
        self.assertEqual([3], split)

    def test_build_multi_input(self):
        mat_in, mat_out = test_data2()

        net = NetworkBuilder(mat_in, mat_out)
        times, indices, split = net.build_input(
                topology_params={"multiplicity": 2},
                input_params=[{"burst_size": 3}, {"burst_size": 2, "isi": 4.0}])

        self.assertEqual([
            [100.0, 101.0, 102.0, 1300.0, 1304.0],
            [100.0, 101.0, 102.0, 1300.0, 1304.0],
            [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 1200.0, 1204.0, 1300.0, 1304.0],
            [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 1200.0, 1204.0, 1300.0, 1304.0],
            [], [],
            [0.0, 1.0, 2.0, 1200.0, 1204.0],
            [0.0, 1.0, 2.0, 1200.0, 1204.0]], times)
        self.assertEqual([
            [1, 1, 1, 3, 3],
            [1, 1, 1, 3, 3],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
            [], [],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]], indices)
        self.assertEqual([2, 4], split)

    def test_build(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build(topology_params={"params": {"cm": 0.2}})
        topo = {
            "connections": net["connections"],
            "populations": net["populations"]
        }
        times = net["input_times"]
        indices = net["input_indices"]
        output_neuron = {'count': 1, 'params':
                {'tau_refrac': 0.1, 'tau_m': 20.0, 'e_rev_E': 0.0,
                'i_offset': 0.0, 'cm': 0.2, 'e_rev_I': -70.0,
                'v_thresh': -50.0, 'tau_syn_E': 5.0, 'v_rest': -65.0,
                'tau_syn_I': 5.0, 'v_reset': -65.0}, 'type':
                'IF_cond_exp', 'record': ['spikes']}
        self.assertEqual({'connections': [
            ((0, 0), (6, 0), 0.03, 0.0),
            ((0, 0), (9, 0), 0.03, 0.0),
            ((1, 0), (6, 0), 0.03, 0.0),
            ((1, 0), (9, 0), 0.03, 0.0),
            ((2, 0), (5, 0), 0.03, 0.0),
            ((2, 0), (6, 0), 0.03, 0.0),
            ((2, 0), (7, 0), 0.03, 0.0),
            ((2, 0), (8, 0), 0.03, 0.0),
            ((3, 0), (5, 0), 0.03, 0.0),
            ((3, 0), (6, 0), 0.03, 0.0),
            ((4, 0), (7, 0), 0.03, 0.0),
            ((4, 0), (8, 0), 0.03, 0.0)], 'populations': [
                {'count': 1, 'params': {'spike_times': [100.0]},
                'type': 'SpikeSourceArray', 'record': []},
                {'count': 1, 'params': {'spike_times': [100.0]},
                'type': 'SpikeSourceArray', 'record': []},
                {'count': 1, 'params': {'spike_times': [0.0, 200.0]},
                'type': 'SpikeSourceArray', 'record': []},
                {'count': 1, 'params': {'spike_times': [200.0]},
                'type': 'SpikeSourceArray', 'record': []},
                {'count': 1, 'params': {'spike_times': [0.0]},
                'type': 'SpikeSourceArray', 'record': []},
                output_neuron, output_neuron, output_neuron, output_neuron,
                output_neuron]}, topo)
        self.assertEqual([[100.0], [100.0], [0.0, 200.0], [200.0], [0.0]],
                times)
        self.assertEqual([[1], [1], [0, 2], [2], [0]], indices)


class TestNetworkInstance(unittest.TestCase):

    def test_match(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build()
        output = [
            {}, {}, {}, {}, {},
            {"spikes": [[0.0, 100.0]]},
            {"spikes": [[101.0]]},
            {"spikes": [[300.0, 301.0]]},
            {"spikes": [[200.0]]},
            {"spikes": [[201.0, 101.0]]}
        ]
        output_spikes, output_indices = net.match(output)
        self.assertEqual([[0.0, 100.0], [101.0], [300.0, 301.0], [200.0],
            [201.0, 101.0]], output_spikes)
        self.assertEqual([[0, 0], [1], [2, 2], [1], [2, 1]], output_indices)

    def test_match_negative(self):
        input_times = [[100.0], [200.0], [300.0], [400.0]]
        input_indices = [[0], [1], [2], [3]]
        output = [
            {}, {}, {}, {},
            {"spikes": [[95.0]]}, {"spikes": [[195.0]]}, {"spikes": [[295.0]]},
            {"spikes": [[395.0]]}
        ]
        output_spikes, output_indices = NetworkInstance.match_static(
                input_times, input_indices, output)
        self.assertEqual(output_spikes, [[95.0], [195.0], [295.0], [395.0]])
        self.assertEqual(output_indices, [[0], [0], [1], [2]])

    def test_split(self):
        times = [[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        indices = [[1, 2, 3, 4, 5], [3, 4, 5, 6], [7, 8, 9]]

        t1, i1 = NetworkInstance.split(times, indices, 2, 5)
        self.assertEqual([[2.0, 3.0, 4.0], [3.0, 4.0], []], t1)
        self.assertEqual([[0, 1, 2], [1, 2], []], i1)

    def test_build_analysis(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build(input_params=[{"multiplicity": 1},
            {"multiplicity": 2}])
        analysis_instances = net.build_analysis(time_mux_output_data)
        test_time_mux_res(self, analysis_instances)


class TestNetworkPool(unittest.TestCase):

    def test_init_net_build_analysis(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        pool = NetworkPool(builder.build(input_params=[{}, {}]))
        analysis_instances = pool.build_analysis(time_mux_output_data)
        test_time_mux_res(self, analysis_instances)

    def test_add_net_build_analysis(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build(input_params=[{}, {}])
        pool = NetworkPool()
        pool.add_network(net)
        analysis_instances = pool.build_analysis(time_mux_output_data)
        test_time_mux_res(self, analysis_instances)

    def test_add_nets_build_analysis(self):
        mat_in, mat_out = test_data()
        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build(input_params=[{"multiplicity": 1},
            {"multiplicity": 2}])
        pool = NetworkPool()
        pool.add_networks([net, net, net])
        analysis_instances = pool.build_analysis(time_mux_output_data * 3)
        self.assertEqual(6, len(analysis_instances))
        test_time_mux_res(self, analysis_instances[0:2])
        test_time_mux_res(self, analysis_instances[2:4])
        test_time_mux_res(self, analysis_instances[4:6])

    def test_preserve_data_params(self):
        dp1 = {
            "n_bits_in": 14,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 3,
            "n_samples": 40
        }
        dp2 = {
            "n_bits_in": 8,
            "n_bits_out": 16,
            "n_ones_in": 2,
            "n_ones_out": 4,
            "n_samples": 50
        }
        net1 = NetworkBuilder(data_params=dp1)
        net2 = NetworkBuilder(data_params=dp2)

        pool = NetworkPool(net1.build())
        pool.add_network(net2.build())

        output = (stub_simulation(net1.mat_out, data_params=dp1) +
                stub_simulation(net2.mat_out, data_params=dp2))

        analysis_instances = pool.build_analysis(output)
        self.assertEqual(len(analysis_instances), 2)
        self.assertEqual(dp1, analysis_instances[0]["data_params"])
        self.assertEqual(dp2, analysis_instances[1]["data_params"])


class TestNetworkAnalysis(unittest.TestCase):

    def test_calculate_latency(self):
        analysis = NetworkAnalysis({
            "input_times": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
            "input_indices": [[0, 1], [2], [3, 4, 5]],
            "output_times": [[1.75, 2.5], [4.5, 4.75], [5.1, 5.2, 6.3], [6.4]],
            "output_indices": [[0, 1], [3, 3], [4, 4, 5], [5]],
            "data_params": {
                "n_bits_in": 3,
                "n_bits_out": 4,
                "n_ones_in": 1,
                "n_ones_out": 1,
                "n_samples": 6
            }
        })
        latency = analysis.calculate_latencies()
        latency[latency == np.inf] = -1 # almost_equal and inf is buggy
        np.testing.assert_almost_equal([0.75, 0.5, -1, 0.75, 0.2, 0.4], latency)

    def test_calculate_latencies(self):
        dp1 = {
            "n_bits_in": 14,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 3,
            "n_samples": 40
        }
        dp2 = {
            "n_bits_in": 8,
            "n_bits_out": 16,
            "n_ones_in": 2,
            "n_ones_out": 4,
            "n_samples": 50
        }
        net1 = NetworkBuilder(data_params=dp1)
        net2 = NetworkBuilder(data_params=dp2)

        pool = NetworkPool(net1.build())
        pool.add_network(net2.build())

        output = (stub_simulation(net1.mat_out, data_params=dp1, latency=20.0) +
                stub_simulation(net2.mat_out, data_params=dp2, latency=30.0))

        analysis_instances = pool.build_analysis(output)
        np.testing.assert_almost_equal(
                [20.0] * dp1["n_samples"],
                analysis_instances[0].calculate_latencies())
        np.testing.assert_almost_equal(
                [30.0] * dp2["n_samples"],
                analysis_instances[1].calculate_latencies())

    def test_calculate_output_matrix(self):
        analysis = NetworkAnalysis({
            "input_times": [[0.0, 0.0], [0.0], [0.0, 0.0, 0.0]],
            "input_indices": [[0, 1], [2], [3, 4, 5]],
            "output_times": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0],
                    [0.0, 0.0, 0.0]],
            "output_indices": [[0, 1, 1, 2], [0, 3, 2], [4, 4], [0, 1, 5]],
            "data_params": {
                "n_bits_in": 3,
                "n_bits_out": 4,
                "n_ones_in": 1,
                "n_ones_out": 1,
                "n_samples": 6
            }
        })

        mat_out = analysis.calculate_output_matrix(
                output_params={"burst_size": 2})
        np.testing.assert_almost_equal([
            [ 0.5,  0.5,  0.,   0.5],
            [ 1.,   0.,   0.,   0.5],
            [ 0.5,  0.5,  0.,   0. ],
            [ 0.,   0.5,  0.,   0. ],
            [ 0.,   0.,   1.,   0. ],
            [ 0.,   0.,   0.,   0.5]], mat_out)

        analysis["topology_params"] = TopologyParameters({
                "multiplicity": 2
        })
        analysis["data_params"]["n_bits_out"] = 2
        mat_out = analysis.calculate_output_matrix(
            output_params={"burst_size": 2})
        np.testing.assert_almost_equal([
            [ 0.5,  0.25],
            [ 0.5,  0.25],
            [ 0.5,  0.],
            [ 0.25, 0.],
            [ 0.,  0.5],
            [ 0.,  0.25]], mat_out)

    def test_calculate_output_matrix_stub_sim(self):
        dp1 = {
            "n_bits_in": 14,
            "n_bits_out": 10,
            "n_ones_in": 3,
            "n_ones_out": 3,
            "n_samples": 40
        }
        dp2 = {
            "n_bits_in": 8,
            "n_bits_out": 16,
            "n_ones_in": 2,
            "n_ones_out": 4,
            "n_samples": 50
        }
        dp3 = {
            "n_bits_in": 5,
            "n_bits_out": 8,
            "n_ones_in": 1,
            "n_ones_out": 3,
            "n_samples": 20
        }
        tp1 = {
            "multiplicity": 1
        }
        tp2 = {
            "multiplicity": 3
        }
        tp3 = {
            "multiplicity": 2
        }

        # Build three networks
        net1 = NetworkBuilder(data_params=dp1)
        net2 = NetworkBuilder(data_params=dp2)
        net3 = NetworkBuilder(data_params=dp3)

        # Add them to a network pool
        pool = NetworkPool()
        pool.add_networks([
                net1.build(topology_params=tp1),
                net2.build(topology_params=tp2),
                net3.build(topology_params=tp3)])

        # Simulate some expected output
        output = (
                stub_simulation(net1.mat_out, data_params=dp1,
                        topology_params=tp1, latency=20.0) +
                stub_simulation(net2.mat_out, data_params=dp2,
                        topology_params=tp2, latency=30.0) +
                stub_simulation(net3.mat_out, data_params=dp3,
                        topology_params=tp3, latency=10.0))

        # Fetch the analysis instance for the output
        analysis_instances = pool.build_analysis(output)

        # Compare the output matrices
        m1 = analysis_instances[0].calculate_output_matrix()
        m2 = analysis_instances[1].calculate_output_matrix()
        m3 = analysis_instances[2].calculate_output_matrix()
        np.testing.assert_almost_equal(net1.mat_out, m1)
        np.testing.assert_almost_equal(net2.mat_out, m2)
        np.testing.assert_almost_equal(net3.mat_out, m3)

