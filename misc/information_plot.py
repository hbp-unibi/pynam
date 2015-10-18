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
Plots the information for an experiment result as created with ./run.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scipy.io as scio
import sys

if len(sys.argv) != 2:
    print "Usage: " + sys.argv[0] + " <TARGET FILE>"
    sys.exit(1)

# Labels for all possible sweep dimensions (wip)
DIM_LABELS = {
    "data.n_bits_in": "Input vector length $m$",
    "data.n_bits_out": "Output vector length $n$",
    "data.n_ones_in": "Number of ones in the input $c$",
    "data.n_ones_out": "Number of ones in the input $d$",
    "data.n_samples": "Number of samples $N$",
    "topology.params.cm": "Membrane capacitance $C_M$ [nF]",
    "topology.params.e_rev_E": "Excitatory reversal potential $E_E$ [mV]",
    "topology.params.e_rev_I": "Inhibitory reversal potential $E_I$ [mV]",
    "topology.params.v_rest": "Resting potential $E_L$ [mV]",
    "topology.params.v_reset": "Reset potential $E_{\\mathrm{reset}}$ [mV]",
    "topology.params.v_thresh": "Threshold potential $E_{\\mathrm{thresh}}$ [mV]",
    "topology.param_noise.cm": "Membrane capacitance noise $\\sigma C_M$ [nF]",
    "topology.param_noise.e_rev_E": "Excitatory reversal potential noise $\\sigma E_E$ [mV]",
    "topology.param_noise.e_rev_I": "Inhibitory reversal potential noise $\\sigma E_I$ [mV]",
    "topology.param_noise.v_rest": "Resting potential noise $\\sigma E_L$ [mV]",
    "topology.param_noise.v_reset": "Reset potential noise $\\sigma E_{\\mathrm{reset}}$ [mV]",
    "topology.param_noise.v_thresh": "Threshold potential noise $\\sigma E_{\\mathrm{thresh}}$ [mV]",
    "topology.multiplicity": "Neuron population size $s$",
    "topology.w": "Synapse weight $w$ [$\\mu$S]",
    "topology.sigma_w": "Synapse weight noise $\\sigma w$ [$\\mu$S]",
    "input.burst_size": "Input burst size $s$",
    "input.time_window": "Time window $T$ [ms]",
    "input.isi": "Burst inter-spike-interval $\Delta t$ [ms]",
    "input.sigma_t": "Spike time noise $\sigma t$ [ms]",
    "input.sigma_t_offs": "Spike time offset noise $\sigma t_{\\mathrm{offs}}$ [ms]",
}

# Labels for all simulators
SIMULATOR_LABELS = {
    "ess": "ESS",
    "nmpm1": "NM-PM-1",
    "nmmc1": "NM-MC-1",
    "spikey": "Spikey",
    "nest": "NEST"
}

# Read the results
results = scio.loadmat(sys.argv[1], squeeze_me=True, struct_as_record=False)

results = {
    "sweep_sigma_t": results
}
#print results

# Iterate over all experiments
for name in results:
    keys = results[name]["keys"]
    dims = results[name]["dims"]
    data = results[name]["data"]

    if dims != 1:
        print "Only one-dimensional experiments are supported (yet), skipping"
        continue

    # Iterate over the data table, find patches with the same key dimensions,
    # create a smaller table containing the mean and standard deviation of all
    # values
    means = np.zeros(data.shape)
    stds = np.zeros(data.shape)
    start = 0
    idx = 0
    while start < data.shape[0]:
        values = data[start, 0:dims]
        end = start + 1
        while end < data.shape[0] and values == data[end, 0:dims]:
            end = end + 1
        means[idx] = np.mean(data[start:end], 0)
        stds[idx] = np.std(data[start:end], 0)

        print stds[idx]

        means[idx, 0:dims] = data[start, 0:dims]
        stds[idx, 0:dims] = data[start, 0:dims]

        idx = idx + 1
        start = end

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = means[0:idx, 0]
    I = means[0:idx, dims]
    I_ref = means[0:idx, dims + 1]
    ax.plot(xs, I, color='k')
    ax.plot(xs, I - stds[0:idx, dims] * 0.5, linestyle=':', color='k')
    ax.plot(xs, I + stds[0:idx, dims] * 0.5, linestyle=':', color='k')
    ax.plot(xs, I_ref, linestyle='--', color='k')

    # Set the simulator label
    if "simulator" in results[name]:
        simulator = results[name]["simulator"]
        if simulator in SIMULATOR_LABELS:
            ax.set_title(SIMULATOR_LABELS[simulator])
        else:
            ax.set_title(simulator)

    # Set the sweep dimension label
    if keys[0] in DIM_LABELS:
        ax.set_xlabel(DIM_LABELS[keys[0]])
    else:
        ax.set_xlabel(keys[0])

    # Set the information label
    ax.set_ylabel("Information $I$ [bit]")

    plt.show()
