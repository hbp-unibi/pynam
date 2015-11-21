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

from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import math
import sys
import os

# Include the PyNAM folder
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), "../pynam"))

import binam
import data
import utils
import entropy

def cm2inch(value):
    return value / 2.54

# Parameters
n_bits = 96
n_ones = 8
n_samples = 1000
n_samples_steps = 10

# Calculate the optimal sample count (used for the )
n_samples_optimal = entropy.optimal_sample_count(n_bits_in = n_bits,
        n_bits_out = n_bits, n_ones_in=n_ones, n_ones_out = n_ones)
print("Optimal sample count for n_bits =", n_bits, "and n_ones =", n_ones, ":",
        n_samples_optimal)

# Generate the input and output data
print("Create input data...")
X = data.generate(n_bits=n_bits, n_ones=n_ones, n_samples=n_samples)
print("Create output data...")
Y = data.generate(n_bits=n_bits, n_ones=n_ones, n_samples=n_samples)

# Train the BiNAM, calculate the error
print("Performing experiment")
M = binam.BiNAM(n_bits, n_bits)
xs = np.array([1] + (np.arange(0, n_samples, n_samples_steps) + n_samples_steps).tolist())
s_old = 0
i = 0

# Initialize the error and information counters
nxs = len(xs)
n_false_positives = np.zeros(nxs)
n_false_negatives = np.zeros(nxs)
n_false_positives_mean = np.zeros(nxs)
n_false_negatives_mean = np.zeros(nxs)
n_false_positives_min = np.zeros(nxs)
n_false_negatives_min = np.zeros(nxs)
n_false_positives_max = np.zeros(nxs)
n_false_negatives_max = np.zeros(nxs)
info = np.zeros(nxs)

for s in xs:
    print("Iteration: ", s)
    # Train the new sample
    for s in xrange(s_old, s):
        M.train(X[s], Y[s])
    s_old = s

    # Evaluate the sample
    X_part = X[0:(s + 1)]
    Y_part = Y[0:(s + 1)]
    Y_part_out = M.evaluate_matrix(X_part)

    # Calculate the errors and the entropy
    errs = entropy.calculate_errs(Y_part_out, Y_part)
    info[i] = entropy.entropy_hetero(errs, n_bits_out=n_bits, n_ones_out=n_ones)
    n_false_positives[i] = np.sum(map(lambda x: x["fp"], errs))
    n_false_negatives[i] = np.sum(map(lambda x: x["fn"], errs))
    n_false_positives_mean[i] = np.mean(map(lambda x: x["fp"], errs))
    n_false_negatives_mean[i] = np.mean(map(lambda x: x["fn"], errs))
    n_false_positives_min[i] = np.min(map(lambda x: x["fp"], errs))
    n_false_negatives_min[i] = np.min(map(lambda x: x["fn"], errs))
    n_false_positives_max[i] = np.max(map(lambda x: x["fp"], errs))
    n_false_negatives_max[i] = np.max(map(lambda x: x["fn"], errs))

    i = i + 1


figsize = (cm2inch(11.8), cm2inch(6))

print("Plotting information...")
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, info, lw=0.75, color="k")
ax.set_xlim(1, n_samples)
ax.set_xlabel("Sample count $N$")
ax.set_ylabel("Information [bits]")

convInfo = entropy.conventional_memory_entropy(n_bits_in=n_bits,
        n_bits_out=n_bits, n_ones_out=n_ones)
ax.plot([0, n_samples], [convInfo, convInfo], lw=0.5, color="k")
anX = 1 + 0.025 * n_samples
ax.annotate(s="\\textit{Conventional memory information}", xy=(anX, convInfo),
        verticalalignment="bottom", horizontalalignment="left", fontsize=8.0)

ax2 = ax.twinx()
ax2.set_xlim(1, n_samples)
ax2.plot(xs,  n_false_positives, '--', lw=0.75, color="#3465a4")

maxEFP = (n_bits - n_ones) * xs
ax2.plot(xs, maxEFP, '--', lw=0.5, color="#3465a4")
ax2.annotate(s="\\textit{Maximum false positives}", xy=(n_samples * 0.95,
        n_samples * (n_bits - n_ones)), verticalalignment="top",
        horizontalalignment="right", fontsize=8.0)

ax2.set_ylabel("Total false positives [bits]", color="#3465a4")
for tl in ax2.get_yticklabels():
    tl.set_color(color="#3465a4")


figsize = (cm2inch(12.8), cm2inch(6))

fig.savefig("out/sketch_info.pdf", format='pdf', bbox_inches='tight')

print("Plotting information per sample...")
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, info / xs, label="Information per sample", lw=0.75, color="k")
ax.plot(xs, n_false_positives_mean, '--', label="False positives per sample", lw=0.75, color="#3465a4")
ax.plot(xs, n_false_positives_min, ':', lw=0.25, color="#3465a4")
ax.plot(xs, n_false_positives_max, ':', lw=0.25, color="#3465a4")

ax.plot([0, n_samples], [n_bits - n_ones, n_bits - n_ones], '--', lw=0.5, color="#3465a4")
ax.annotate(s="\\textit{Maximum false positives}", xy=(n_samples * 0.975, n_bits - n_ones),
        verticalalignment="bottom", horizontalalignment="right", fontsize=8.0)

mInfo = entropy.lnncrr(n_bits, n_ones) / math.log(2.0)
ax.plot([0, n_samples], [mInfo, mInfo], '-', lw=0.5, color="k")
ax.annotate(s="\\textit{Maximum information}", xy=(n_samples * 0.975, mInfo),
        verticalalignment="bottom", horizontalalignment="right", fontsize=8.0)


ax.set_xlim(1, n_samples)
ax.set_xlabel("Sample count $N$")
ax.set_ylabel("Bits")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
fig.savefig("out/sketch_info_per_sample.pdf", format='pdf', bbox_inches='tight')

print("Plotting errors...")
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, n_false_positives_mean, label="False positives", color="#3465a4")
ax.plot(xs, n_false_negatives_mean, label="False negatives", color="#75507b")
ax.plot(xs, n_false_positives_min, ':', lw=0.5, color="#3465a4")
ax.plot(xs, n_false_negatives_min, ':', lw=0.5, color="#75507b")
ax.plot(xs, n_false_positives_max, ':', lw=0.5, color="#3465a4")
ax.plot(xs, n_false_negatives_max, ':', lw=0.5, color="#75507b")

ax.plot([0, n_samples], [n_bits - n_ones, n_bits - n_ones], '--', lw=0.5, color="#3465a4")
ax.annotate(s="\\textit{Maximum false positives}", xy=(anX, n_bits - n_ones),
        verticalalignment="top", horizontalalignment="left", fontsize=8.0)

ax.plot([0, n_samples], [n_ones, n_ones], '--', lw=0.5, color="#75507b")
ax.annotate(s="\\textit{Maximum false negatives}", xy=(anX, n_ones),
        verticalalignment="bottom", horizontalalignment="left", fontsize=8.0)


ax.set_xlim(1, n_samples)
ax.set_xlabel("Sample count $N$")
ax.set_ylabel("Average error bits per sample")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
fig.savefig("out/sketch_errs.pdf", format='pdf', bbox_inches='tight')

