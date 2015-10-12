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
        NetworkInstance)

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

    def test_build_topology(self):
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
        mat_in = BinaryMatrix(3, 5)
        mat_in[0, 4] = 1
        mat_in[0, 2] = 1
        mat_in[1, 0] = 1
        mat_in[1, 1] = 1
        mat_in[2, 3] = 1
        mat_in[2, 2] = 1

        mat_out = BinaryMatrix(3, 5)

        net = NetworkBuilder(mat_in, mat_out)
        times, indices = net.build_input()
        self.assertEqual([[100.0], [100.0], [0.0, 200.0], [200.0], [0.0]], times)
        self.assertEqual([[1], [1], [0, 2], [2], [0]], indices)

        times, indices = net.build_input(topology_params={"multiplicity": 2})
        self.assertEqual([[100.0], [100.0], [100.0], [100.0], [0.0, 200.0],
                [0.0, 200.0], [200.0], [200.0], [0.0], [0.0]], times)
        self.assertEqual([[1], [1], [1], [1], [0, 2], [0, 2],
                [2], [2], [0], [0]], indices)

        times, indices = net.build_input(topology_params={"multiplicity": 2},
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

    def test_build(self):
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

        builder = NetworkBuilder(mat_in, mat_out)
        net = builder.build()
        topo = {
            "connections": net["connections"],
            "populations": net["populations"]
        }
        times = net["input_times"]
        indices = net["input_indices"]
        output_neuron = {'count': 1, 'params':
                {'tau_refrac': 0.1, 'tau_m': 20.0, 'e_rev_E': 0.0,
                'i_offset': 0.0, 'cm': 1.0, 'e_rev_I': -70.0,
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

