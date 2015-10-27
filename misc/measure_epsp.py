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
Used to record and measure the membrane potential for the specified neuron
parameters.
"""

import os
import sys

import numpy as np
import scipy.io as scio

# Include PyNNLess
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__),
        "../lib/pynnless"))
import pynnless as pynl

# Used neuron parameters
params = {
    "cm": 0.2,
    "e_rev_E": 0,
    "v_rest": -70,
    "v_reset": -75,
    "v_thresh": -55,
    "tau_m": 10.0
}

# Issue exactly one spike at t = 1000.0
spike_times = [1000.0]

# Number of neurons for which the simulation should be performed
neuron_count = 192

weight = 0.008

# Target simulator
backend = "spikey"

# Record from each neuron exactly once
vs = [[]] * neuron_count
ts = [[]] * neuron_count
for i in xrange(neuron_count):
    print "Measuring data for neuron", i
    res = pynl.PyNNLessIsolated(backend).run(pynl.Network()
                .add_source(spike_times = spike_times)
                .add_populations([{
                        "type": pynl.TYPE_IF_COND_EXP,
                        "params": params,
                        "record": [pynl.SIG_V] if i == j else []
                    } for j in xrange(neuron_count)])
                .add_connections([((0, 0), (j + 1, 0), weight, 0.0)
                    for j in xrange(neuron_count)]))
    vs[i] = res[i + 1]["v"][0]
    ts[i] = res[i + 1]["v_t"]

# Calculate the compound sample times
ts_res = np.arange(0.0, 2000.0, 0.1)
vs_res = np.zeros((neuron_count, len(ts_res)))
for i in xrange(neuron_count):
    vs_res[i] = np.interp(ts_res, ts[i], vs[i])

# Create a version with corrected DC-offset
vs_offs = np.zeros((neuron_count, len(ts_res)))
for i in xrange(neuron_count):
    vs_offs[i] = (vs_res[i] - np.mean(np.sort(vs_res[i])[:int(-len(ts) * 0.75)])
            + params["v_rest"])

# Calculate the average
vs_avg = np.mean(vs_offs, 0)

# Store the two matrices in the resulting matlab file
scio.savemat("out/" + backend + "_epsp.mat", {
    "ts": ts_res,
    "vs": vs_res,
    "vs_offs": vs_offs,
    "vs_avg": vs_avg
})

# Store the CSV version of the data
M = np.vstack((ts_res, vs_avg)).T
np.savetxt("out/" + backend + "_epsp.csv", M, delimiter=",")
