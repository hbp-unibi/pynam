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

import pynam

