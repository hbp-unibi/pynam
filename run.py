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
Main program of the PyNAM framework. This program can operate in different
modes: It can run a complete experiment, including network creation, execution
and analysis.
"""

import gzip
import pickle

import os.path
import time
import datetime

import logging
import subprocess

import numpy as np
import pynam
import pynam.entropy
import pynnless as pynl
import scipy.io as scio
import sys

# Get a logger, write to stderr
logger = logging.getLogger("PyNAM")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def help():
    print("""
PyNAM -- Python Neural Associative Memory Simulator and Evaluator

Copyright (C) 2015 Andreas Stöckel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


Usage:

To run the complete analysis pipeline, run the program in one of the following
forms:

    ./run.py <SIMULATOR> [<EXPERIMENT>]
    ./run.py <SIMULATOR> --process <EXPERIMENT>

Where <SIMULATOR> is the simulator that should be used for execution (see below)
and <EXPERIMENT> is a JSON file describing the experiment that should be executed.
If <EXPERIMENT> is not given, the program will try to execute "experiment.json".

In order to just generate the network descriptions for a specific experiment,
use the following format:

    ./run.py <SIMULATOR> --create <EXPERIMENT>

Note that the <SIMULATOR> is needed in order to partition the experiment
according to the hardware resources. This command will generate a set of
".in.json.gz" files that can be passed to the execution stage:

    ./run.py <SIMULATOR> --exec <IN_1> ... <IN_N>

This command will execute the network simulations for the given network
descriptions -- if more than one network descriptions is given, a new process
will be spawned for each execution. Generates a series of ".out.json.gz" files
in the same directory as the input files.

    ./run.py <TARGET> --analyse <OUT_1> ... <OUT_N>

Analyses the given output files, generates a HDF5/Matlab file as <TARGET>
containing the processed results.
""")
    print("<SIMULATOR> may be one of the following, these simulators have been")
    print("auto-detected on your system:")
    print("\t" + str(pynl.PyNNLess.simulators()))
    sys.exit(1)

def short_help():
    print("Type\n\t./run.py --help\nto get usage information.")
    sys.exit(1)

def parse_parameters():
    # Error case: Need at least one argument
    if len(sys.argv) == 1:
        print("Error: At least one argument is required")
        short_help()

    # Make sure the first argument does not start with "--"
    if len(sys.argv) > 2 and sys.argv[1].startswith("--"):
        if (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
            help()
        print("Error: Invalid arguments")
        short_help()

    # Special case -- only one argument is given -- print help if the first
    # argument is "-h" or "--help"
    if len(sys.argv) == 2:
        return {
            "mode": "process",
            "simulator": sys.argv[1],
            "experiment": "experiment.json"
        }


    # Special case two: Three parameters are given and the third parameter does
    # not start with "--"
    if len(sys.argv) == 3 and not sys.argv[2].startswith("--"):
        return {
            "mode": "process",
            "simulator": sys.argv[1],
            "experiment": sys.argv[2]
        }

    # We need at least three parameters and the second parameter needs to be
    # in ["process", "create", "exec", "analyse"]
    if len(sys.argv) < 3 or not sys.argv[2].startswith("--"):
        print("Error: Invalid arguments")
        short_help()
    mode = sys.argv[2][2:]
    if not mode in ["process", "create", "exec", "analyse"]:
        print("Error: Processing mode must be one of [process, create, exec, "
                + "analyse], but \"" + mode + "\" was given")
        short_help()

    # Require at least one argument
    if len(sys.argv) < 4:
        print("Error: At least one argument is required for mode \"" + mode  +
                "\"")
        short_help()

    # Parse the "process" mode
    if mode == "process" or mode == "create":
        if len(sys.argv) != 4:
            print("Error: Mode \"" + mode + "\" requires exactly one argument")
            short_help()
        else:
            return {
                "mode": mode,
                "simulator": sys.argv[1],
                "experiment": sys.argv[3]
            }

    # Parse the "create" and "exec" modes
    if mode == "exec":
        return {
            "mode": mode,
            "simulator": sys.argv[1],
            "files": sys.argv[3:]
        }

    # Parse the "analyse" mode
    if mode == "analyse":
        return {
            "mode": "analyse",
            "target": sys.argv[1],
            "files": sys.argv[3:]
        }

    # Something went wrong, print the usage help
    short_help()

def validate_parameters(params):
    # Make sure the specified input files exist
    files = []
    if "experiment" in params:
        files.append(params["experiment"])
    if "files" in params:
        files = files + params["files"]
    for fn in files:
        if not os.path.isfile(fn):
            print "Error: Specified file \"" + fn + "\" does not exist."
            short_help()

def read_experiment(experiment_file):
    return pynam.Experiment.read_from_file(experiment_file)

def write_compressed(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_compressed(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def create_networks(experiment_file, simulator, path=""):
    """
    Create the network descriptions and write them to disc. Returns the names
    of the created files, separated by experiment
    """

    # Read the experiment
    experiment = read_experiment(experiment_file)

    # Build the experiment descriptors
    seed = 1437243
    logger.info("Generating networks...")
    pools = experiment.build(pynl.PyNNLess.get_simulator_info_static(simulator),
            seed)

    # Create the target directories
    if path != "":
        os.makedirs(path)

    # Store the descriptors in the given path
    input_files = []
    output_files = []
    for pool in pools:
        in_filename = os.path.join(path, pool["name"] + ".in.pickle.gz")
        out_filename = os.path.join(path, pool["name"] + ".out.pickle.gz")

        logger.info("Writing network descriptor to: " + in_filename)
        write_compressed(pool, in_filename)

        input_files.append(in_filename)
        output_files.append(out_filename)
    return input_files, output_files

def execute_networks(input_files, simulator):
    # If there is more than one input_file, execute these in different processes
    if (len(input_files) > 1):
        # Fetch the concurrency supported by the simulator
        concurrency = (pynl.PyNNLess.get_simulator_info_static(simulator)
                ["concurrency"])

        # Assemble the processes that should be executed
        cmds = [[sys.argv[0], simulator, "--exec", x] for x in input_files]
        processes = []
        had_error = False
        while (((len(cmds) > 0) or (len(processes) > 0))
                and not (len(processes) == 0 and had_error)):
            # Spawn new processes
            if ((len(processes) < concurrency) and (not had_error)
                    and (len(cmds) > 0)):
                cmd = cmds.pop()
                logger.info("Executing " + " ".join(cmd))
                processes.append(subprocess.Popen(cmd))

            # Check whether any of the processes has finished
            for i, process in enumerate(processes):
                if not process.poll() is None:
                    if process.returncode != 0:
                        had_error = True
                    del processes[i]
                    continue

            # Sleep a short while before rechecking
            time.sleep(0.1)

        # Return whether the was an error during execution
        if had_error:
            logger.error("There was an error during network execution!")
        return had_error

    # Fetch the input file
    input_file = input_files[0]

    # Generate the output file name
    if (input_file.endswith(".in.pickle.gz")):
        output_file = input_file[:-12] + "out.pickle.gz"
    else:
        output_file = input_file + ".out.pickle.gz"

    # Read the input file
    logger.info("Reading input file: " + input_file)
    input_network = read_compressed(input_file)

    logger.info("Setup simulator...")
    sim = pynl.PyNNLess(simulator)

    logger.info("Run simulation...")
    output = sim.run(input_network)

    logger.info("Writing output file: " + output_file)
    write_compressed({
        "input": input_network,
        "times": sim.get_time_info(),
        "output": output
    }, output_file)

    return True

#
# Main entry point -- parse and validate the parameters and depending on those,
# execute the corresponding functions defined above
#

# Parse the parameters and validate them
params = parse_parameters()
validate_parameters(params)

if params["mode"] == "create":
    create_networks(params["experiment"], params["simulator"], "")
elif params["mode"] == "exec":
    if not execute_networks(params["files"], params["simulator"]):
        sys.exit(1)
elif params["mode"] == "process":
    # Assemble a directory for the experiment files
    folder = os.path.join("out",
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + "_"
                    + params["simulator"])

    # Create the networks and fetch the input and output files
    input_files, output_files = create_networks(
            params["experiment"], params["simulator"], folder)

    # Execute the networks, abort if the execution failed
    if not execute_networks(input_files, params["simulator"]):
        sys.exit(1)

logger.info("Done.")
sys.exit(0)

# Build the network and the metadata
#print "Build network..."

# Run the simulations and print the analysis results
#for pool in pools:
#    print "Running ", pool["name"], "..."
#    output = sim.run(pool)
#    print "Analyzing data..."
#    analysis_instances = pool.build_analysis(output)
#    for analysis in analysis_instances:
#        I_ref, mat_ref, errs_ref = analysis.calculate_max_storage_capacity()
#        I, mat, errs = analysis.calculate_storage_capactiy()
#        fp_ref = sum(map(lambda x: x["fp"], errs_ref))
#        fn_ref = sum(map(lambda x: x["fn"], errs_ref))
#        fp = sum(map(lambda x: x["fp"], errs))
#        fn = sum(map(lambda x: x["fn"], errs))
#        print\
#            analysis["input_params"]["sigma_t"], ",",\
#            analysis["topology_params"]["w"], ",",\
#            I, ",", I_ref, ",", fp, ",", fp_ref, ",", fn, ",", fn_ref

