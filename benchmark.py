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

import datetime
import json
import os
import pynam.utils
import sys
import run as model

if len(sys.argv) != 3:
    print sys.argv[0] + " <SYSTEM> <EXPERIMENT>"
    sys.exit(1)

simulator = sys.argv[1]
experiment = sys.argv[2]
global_experiment_name = os.path.basename(os.path.splitext(experiment)[0])

folder = os.path.join("out", "benchmark")

# Create the networks and fetch the input and output files
input_files, output_files = model.create_networks(experiment, simulator, folder, analyse=True)
if not model.execute_networks(input_files, simulator, analyse=True):
    sys.exit(1)

analysis = {}
for output_file in output_files:
    analysis_part = pynam.utils.loadmat(output_file)
    analysis = model.append_analysis(analysis, analysis_part)
analysed_results = model.finalize_analysis(analysis)

# Remove the partial input and output files (no longer needed)
for input_file in input_files:
    try:
        os.remove(input_file)
    except:
        logger.warn("Error while deleting " + input_file)
for output_file in output_files:
    try:
        os.remove(output_file)
    except:
        logger.warn("Error while deleting " + output_file)
try:
    os.rmdir(folder)
except:
    logger.warn("Error while deleting " + folder)

# Write the output JSON file containing the benchmark results
benchmark_results = {
    "time": datetime.datetime.now().isoformat(),
    "results": []
}
experiments = analysed_results.keys()
experiments.sort()
for experiment in experiments:
    results = analysed_results[experiment]
    keys = map(lambda s: s.strip(), results["keys"].tolist())
    idx = keys.index("I_n")
    benchmark_results["results"].append({
        "type": "information",  # should probably be one of "quality", "performance",
        "name": global_experiment_name + "__" + experiment,
        "value": results["data"][0][idx],
        "measure": "unknown"  # to be implemented
    })

with open('benchmark_results.json', 'w') as outfile:
    json.dump(benchmark_results, outfile, indent=4, sort_keys=True)

