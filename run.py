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

# The "run.py" script is the main script of PyNAM. It can operate in two
# different modes:
#
#   * The "experiment mode". In this case the name of the PyNN backend is passed
#     as a single parameter. A configuration file will be read from the
#     "config.json" file. Optionally, the name of the configuration file can
#     be passed as a second parameter. The experiment is splitted into a list
#     of simulations that can run concurrently. Writes the results to a Matlab
#     file "result.mat"
#
#     Examples:
#         ./run.py nest
#         ./run.py nest experiment.json experiment.mat
#
#   * The "simulation mode". Performs a single simulation run. Reads the
#     simulation parameters from stdin, writes the result to stdout. Both stdin
#     and stdout are in JSON format. Simulation mode is activated by passing the
#     single "--simulation" parameter.
#
#     Examples:
#         cat config.json | ./run.py --simulation > result.json

import numpy as np
import pynam
import pynam.binam_utils
import pynnless as pynl
import scipy.io as scio
import sys

if (len(sys.argv) != 2):
    print("Usage: " + sys.argv[0] + " <SIMULATOR>")
    sys.exit(1)

# Generate test data
print "Generate test data..."
m = 8
n = 8
c = 3
d = 3
N = 10

topology_params = {
    "w": 0.016,
    "params": {
        "cm": 0.2,
        "e_rev_E": -40,
        "e_rev_I": -60,
        "v_rest": -50,
        "v_reset": -70,
        "v_thresh": -47,
        "tau_m": 409.0,
        "tau_refrac": 20.0
    }
}

input_params = {
    "time_window": 500.0,
    "sigma_t": 5.0
}

mat_in = pynam.generate(n_bits=m, n_ones=c, n_samples=N)
mat_out_expected = pynam.generate(n_bits=n, n_ones=d, n_samples=N)

# Train a reference binam
binam = pynam.BiNAM(m, n)
binam.train_matrix(mat_in, mat_out_expected)
mat_out_ref = binam.evaluate_matrix(mat_in)
errs_ref = pynam.binam_utils.calculate_errs(mat_out_ref, mat_out_expected)
I_ref = pynam.binam_utils.entropy_hetero(errs_ref, n, d)

# Build the network and the metadata
print "Build network..."
builder = pynam.NetworkBuilder(mat_in, mat_out_expected)
net = builder.build(topology_params=topology_params, input_params=input_params)

# Run the simulation
print "Initialize simulator..."
sim = pynl.PyNNLess(sys.argv[1])
print "Run simulation..."
output = sim.run(net)

print "OUTPUT:\n", output

# Fetch the output times and output indices from the output data
print "Analyze result..."
analysis = net.build_analysis(output)[0]
I, mat_out, errs = analysis.calculate_storage_capactiy(mat_out_expected, d,
        topology_params=topology_params)
latency = analysis.calculate_latencies()

print "INFORMATION: ", I, " of a theoretical ", I_ref
print "MAT OUT:\n", mat_out
print "MAT OUT (reference):\n", mat_out_ref
print "MAT OUT EXPECTED:\n", mat_out_expected
print "AVG. LATENCY:\n", np.mean(latency)
print "LATENCIES:\n", latency

scio.savemat("res.mat", {
    "mat_in": mat_in,
    "mat_out_expected": mat_out,
    "mat_out_ref": mat_out_ref,
    "mat_out": mat_out,
    "errs": errs,
    "errs_ref": errs_ref,
    "I": I,
    "I_ref": I_ref,
    "latency": latency,
    "topology_params": topology_params,
    "input_params": input_params,
    "m": m,
    "n": n,
    "c": c,
    "d": d,
    "N": N
})

