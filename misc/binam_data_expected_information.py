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
import binam_data
import binam_utils

# Parameters
n_bits = [16, 16, 16, 32, 32, 32]
n_ones = [2, 3, 4, 2, 3, 4]
n_samples = 200
n_experiments = len(n_bits)

def cm2inch(value):
    return value/2.54

colors = [
    '#47294F',
    '#3C7705',
    '#800000',
    '#193A6B',
    '#A08300',
    '#AA4C00',
]
styles = [
    '-', '--', ':'
]

info = np.zeros((n_experiments, n_samples + 1))
errs = np.zeros((n_experiments, n_samples + 1))
t = range(0, n_samples + 1)
for i in xrange(n_experiments):
    c = n_ones[i]
    d = n_ones[i]
    m = n_bits[i]
    n = n_bits[i]
    for N in t:
        err = binam_utils.expected_false_positives(N, n, d, m, c)
        errs[i, N] = err
        info[i, N] = binam_utils.entropy_hetero_uniform(err, N, m, d)

fig1 = plt.figure(figsize=(cm2inch(5), cm2inch(5)))
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('Number of trained samples $N$')
ax1.set_ylabel('Information $I$ [Bit]')
ax1.set_xlim(0, n_samples)
ax1.set_ylim(0, np.max(info))

fig2 = plt.figure(figsize=(cm2inch(5), cm2inch(5)))
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Number of trained samples $N$')
ax2.set_ylabel('Expected false positives $\\langle n_{\\mathrm{fp}} \\rangle$')
ax2.set_xlim(0, n_samples)
ax2.set_ylim(0, np.max(errs))

for i in xrange(n_experiments):
    color = colors[i / 3]
    style = styles[i % 3]
    label = "$m = n = " + str(n_bits[i]) + ", c = d = " + str(n_ones[i]) + "$"
    ax1.plot(t, info[i], color=color, linestyle=style, label=label)
    ax2.plot(t, errs[i], color=color, linestyle=style, label=label)

#    b = binam_utils.conventional_memory_entropy(n_bits[i], n_bits[i], n_ones[i])
#    ax1.plot([0, n_samples], [b, b], linestyle=style, color=color, zorder=-1)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)

fig1.savefig("out/expected_info.pdf", format='pdf', bbox_inches='tight')
fig2.savefig("out/expected_err.pdf", format='pdf', bbox_inches='tight')

plt.show()

