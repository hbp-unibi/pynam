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

# Include the PyNAM folder
import sys
import os
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), "../pynam"))

import binam
import binam_data

# Possible sizes (for which reference data has been generated)
size = 128
#size = 256
nBits = 3
nSamples = 10000

def cm2inch(value):
    return value/2.54

# Calculate the correllation matrices and plot both
def plot(m, title):
    print "Calculating correlation for \"", title, "\""
    C = np.corrcoef(m.T)
    print "Plotting..."

    fig = plt.figure(figsize=(cm2inch(7.5), cm2inch(6)))
    ax = fig.add_subplot("111")
    ax.set_xlabel("Bit index $i$")
    ax.set_ylabel("Bit index $j$")
    ax.set_title(title)
    cax = ax.imshow(C, interpolation="none", vmin=0, vmax=0.1,
            cmap="Blues")
    fig.colorbar(cax, ticks=[0, 0.05, 0.1])
    return fig

# Plot the own data
print "Generate own data...."
plot(binam_data.generate(size, nBits, nSamples),
    "With selection bias")\
    .savefig("out/balanced_with_bias.pdf", format='pdf', bbox_inches='tight')

print "Generate own data (with weight_choices=False)..."
plot(binam_data.generate(size, nBits, nSamples, weight_choices=False),
    "Without selection bias")\
    .savefig("out/balanced_no_bias.pdf", format='pdf', bbox_inches='tight')

#print "Generate own data (naive method)..."
#plot(binam_data.generate_naive(size, 3, 10000),
#    "Python generator (naive)")

plt.show()
