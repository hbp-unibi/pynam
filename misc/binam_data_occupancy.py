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
Visualizes the occupancy distribution of the BiNAM depending on the data
generation method.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Include the PyNAM folder
import sys
import os
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), "../pynam"))

import binam
import binam_data

n_bits = 32
n_ones = 16
n_samples = 100
n_it = 1000

# Calculate the occupancy of a BiNAM-Matrix
results = {"data": [], "title": []}
def calculate(func, title):
    global results
    print "Generating data for \"" + title + "\""
    n = n_bits * n_bits
    vs = np.zeros(n_it * n, dtype=np.int32)
    for i in xrange(n_it):
        sys.stdout.write("Generation " + str(i + 1) + " of " + str(n_it) + "\r")
        sys.stdout.flush()
        m1 = func()
        m2 = func()
        vs[(i*n):((i+1)*n)] = np.reshape(np.dot(m1.T, m2), (n))
    results["data"].append(vs)
    results["title"].append(title)
    print

def cm2inch(value):
    return value/2.54

def plot():
    fig = plt.figure(figsize=(cm2inch(13), cm2inch(6)))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(results["data"])
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color='black')
    ax.set_xticklabels(results["title"])
    ax.set_ylim(0, np.max(results["data"]))
    ax.set_ylabel("Occupancy")
    plt.savefig('out/binam_occupancy_data.pdf', format='pdf',
        bbox_inches='tight')

calculate(lambda: binam_data.generate_random(n_bits, n_ones, n_samples),
    "Random\n(with duplicates)")
calculate(lambda: binam_data.generate(n_bits, n_ones, n_samples, balance=False),
    "Random\n(no duplicates)")
calculate(lambda: binam_data.generate_naive(n_bits, n_ones, n_samples),
    "Balanced\n(with duplicates)")
calculate(lambda: binam_data.generate(n_bits, n_ones, n_samples),
    "Balanced\n(no duplicates)")
plot()

