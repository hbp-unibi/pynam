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
import pynam.entropy
import pynnless as pynl
import scipy.io as scio
import sys

if (len(sys.argv) != 2):
    print("Usage: " + sys.argv[0] + " <SIMULATOR> [<EXPERIMENT>]")
    print("       " + sys.argv[0] + " <SIMULATOR> --create [<EXPERIMENT>]")
    print("       " + sys.argv[0] + " <SIMULATOR> --run <POOL_1> ... <POOL_N>")
    print("       " + sys.argv[0] + " <SIMULATOR> --gather <OUT_1> ... <OUT_N>")
    sys.exit(1)

# Build the network and the metadata
print "Build network..."

experiment = pynam.Experiment.read_from_file("experiment.json")

seed = 1437243
sim = pynl.PyNNLess(sys.argv[1])
pools = experiment.build(sim.get_simulator_info(), seed)

# Run the simulations and print the analysis results
for pool in pools:
    print "Running ", pool["name"], "..."
    output = sim.run(pool)
    print "Analyzing data..."
    analysis_instances = pool.build_analysis(output)
    for analysis in analysis_instances:
        I_ref, mat_ref, errs_ref = analysis.calculate_max_storage_capacity()
        I, mat, errs = analysis.calculate_storage_capactiy()
        fp_ref = sum(map(lambda x: x["fp"], errs_ref))
        fn_ref = sum(map(lambda x: x["fn"], errs_ref))
        fp = sum(map(lambda x: x["fp"], errs))
        fn = sum(map(lambda x: x["fn"], errs))
        print\
            analysis["input_params"]["sigma_t"], ",",\
            analysis["topology_params"]["w"], ",",\
            I, ",", I_ref, ",", fp, ",", fp_ref, ",", fn, ",", fn_ref

