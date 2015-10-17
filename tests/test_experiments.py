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

import collections
import unittest

import numpy as np
import numpy.testing

from pynam.experiment import (ExperimentException, Experiment,
        ExperimentDescriptor, ExperimentSweep)

class TestExperimentDescriptor(unittest.TestCase):

    def test_build_combinatorial_sweep_vectors(self):
        sweeps = collections.OrderedDict([
            ("a", [1, 2, 3, 4]),
            ("aa", []),
            ("c", [7, 8, 9]),
            ("b", [5, 6]),
            ("d", [])
        ])

        keys, vecs = ExperimentDescriptor.build_combinatorial_sweep_vectors(
                sweeps)
        self.assertEqual(keys, ["a", "c", "b"])
        numpy.testing.assert_almost_equal(vecs, [
                [ 1.,  7.,  5.],
                [ 2.,  7.,  5.],
                [ 3.,  7.,  5.],
                [ 4.,  7.,  5.],
                [ 1.,  8.,  5.],
                [ 2.,  8.,  5.],
                [ 3.,  8.,  5.],
                [ 4.,  8.,  5.],
                [ 1.,  9.,  5.],
                [ 2.,  9.,  5.],
                [ 3.,  9.,  5.],
                [ 4.,  9.,  5.],
                [ 1.,  7.,  6.],
                [ 2.,  7.,  6.],
                [ 3.,  7.,  6.],
                [ 4.,  7.,  6.],
                [ 1.,  8.,  6.],
                [ 2.,  8.,  6.],
                [ 3.,  8.,  6.],
                [ 4.,  8.,  6.],
                [ 1.,  9.,  6.],
                [ 2.,  9.,  6.],
                [ 3.,  9.,  6.],
                [ 4.,  9.,  6.],])

    def test_validate_keys(self):
        Experiment.validate_keys(["input.sigma_t", "input.p0"])
        Experiment.validate_keys(["topology.w"])
        Experiment.validate_keys(["topology.params.tau_m"])
        Experiment.validate_keys(["topology.param_noise.tau_m"])
        Experiment.validate_keys(["data.n_ones_in"])
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys([""]))
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys(["bla"]))
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys(["bla.blub"]))
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys(["input"]))
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys(["input.blub"]))
        self.assertRaises(ExperimentException,
                lambda: Experiment.validate_keys(["input.sigma_t.test"]))

    def test_build_parameters(self):
        experiment = Experiment()

        input_params, topology_params = experiment.build_parameters(
                ExperimentDescriptor(sweeps={
                    "input.sigma_t": [0.0, 1.0, 2.0],
                }))
        self.assertEqual(len(input_params), 3)
        self.assertEqual(len(topology_params), 1)
        self.assertEqual(input_params[0]["sigma_t"], 0.0)
        self.assertEqual(input_params[1]["sigma_t"], 1.0)
        self.assertEqual(input_params[2]["sigma_t"], 2.0)


        input_params, topology_params = experiment.build_parameters(
                ExperimentDescriptor(sweeps={
                    "input.sigma_t": {"min": 0.0, "max": 2.0, "count": 3}
                }))
        self.assertEqual(len(input_params), 3)
        self.assertEqual(len(topology_params), 1)
        self.assertEqual(input_params[0]["sigma_t"], 0.0)
        self.assertEqual(input_params[1]["sigma_t"], 1.0)
        self.assertEqual(input_params[2]["sigma_t"], 2.0)

        input_params, topology_params = experiment.build_parameters(
                ExperimentDescriptor(sweeps={
                    "topology.params.tau_m": [0.0, 1.0, 2.0],
                }))
        self.assertEqual(len(input_params), 1)
        self.assertEqual(len(topology_params), 3)
        self.assertEqual(topology_params[0]["topology"]["params"]["tau_m"], 0.0)
        self.assertEqual(topology_params[1]["topology"]["params"]["tau_m"], 1.0)
        self.assertEqual(topology_params[2]["topology"]["params"]["tau_m"], 2.0)

        input_params, topology_params = experiment.build_parameters(
                ExperimentDescriptor(sweeps={
                    "topology.param_noise.tau_m": [0.0, 1.0, 2.0],
                }))
        self.assertEqual(len(input_params), 1)
        self.assertEqual(len(topology_params), 3)
        self.assertEqual(topology_params[0]["topology"]["param_noise"]["tau_m"],
                0.0)
        self.assertEqual(topology_params[1]["topology"]["param_noise"]["tau_m"],
                1.0)
        self.assertEqual(topology_params[2]["topology"]["param_noise"]["tau_m"],
                2.0)

        input_params, topology_params = experiment.build_parameters(
                ExperimentDescriptor(sweeps=collections.OrderedDict([
                    ("topology.param_noise.tau_m", [0.0, 1.0, 2.0]),
                    ("data.n_ones_in", [1, 2])
                ])))
        self.assertEqual(len(input_params), 1)
        self.assertEqual(len(topology_params), 6)
        self.assertEqual(topology_params[0]["topology"]["param_noise"]["tau_m"],
                0.0)
        self.assertEqual(topology_params[1]["topology"]["param_noise"]["tau_m"],
                1.0)
        self.assertEqual(topology_params[2]["topology"]["param_noise"]["tau_m"],
                2.0)
        self.assertEqual(topology_params[3]["topology"]["param_noise"]["tau_m"],
                0.0)
        self.assertEqual(topology_params[4]["topology"]["param_noise"]["tau_m"],
                1.0)
        self.assertEqual(topology_params[5]["topology"]["param_noise"]["tau_m"],
                2.0)
        self.assertEqual(topology_params[0]["data"]["n_ones_in"], 1)
        self.assertEqual(topology_params[1]["data"]["n_ones_in"], 1)
        self.assertEqual(topology_params[2]["data"]["n_ones_in"], 1)
        self.assertEqual(topology_params[3]["data"]["n_ones_in"], 2)
        self.assertEqual(topology_params[4]["data"]["n_ones_in"], 2)
        self.assertEqual(topology_params[5]["data"]["n_ones_in"], 2)

