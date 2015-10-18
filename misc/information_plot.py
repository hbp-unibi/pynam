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
import sys

# Include the PyNAM folder
import sys
import os
sys.path.append(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pynam.utils

if len(sys.argv) < 2:
    print "Usage: " + sys.argv[0] + " <TARGET FILE 1> ... <TARGET FILE N>"
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
    "nmpm1": "NM-PM1",
    "nmmc1": "NM-MC1",
    "spikey": "Spikey",
    "nest": "NEST"
}

# Colors for all simulators
SIMULATOR_COLORS = {
    "ess": '#73d216',
    "nmpm1": "#f57900",
    "nmmc1": "#75507b",
    "spikey": "#cc0000",
    "nest": "#3465a4"
}

figures = {}

def cm2inch(value):
    return value / 2.54

def get_figure(experiment, measure, simulator):
    global figures
    first = False
    if not experiment in figures:
        figures[experiment] = {}
    if not measure in figures[experiment]:
        first = True
        figures[experiment][measure] = {}
        fig = plt.figure(figsize=(cm2inch(8), cm2inch(6)))
        ax = fig.add_subplot(111)
        figures[experiment][measure]["figure"] = fig
        figures[experiment][measure]["axis"] = ax
        figures[experiment][measure]["count"] = 0
        figures[experiment][measure]["simulators"] = []
    figures[experiment][measure]["count"] =\
            figures[experiment][measure]["count"] + 1
    figures[experiment][measure]["simulators"].append(simulator)
    return figures[experiment][measure]["axis"], first,\
            figures[experiment][measure]["count"]

def calc_means_stds(data, dims):
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

        means[idx, 0:dims] = data[start, 0:dims]
        stds[idx, 0:dims] = data[start, 0:dims]

        idx = idx + 1
        start = end

    means.resize((idx, data.shape[1]))
    stds.resize((idx, data.shape[1]))

    return means, stds

def plot_measure(ax, xs, ys, ys_std, color, simulator, xlabel, ylabel,
        ys_ref=None, first=True):
    if first:
        if not ys_ref is None:
            ax.plot(xs, ys_ref, linestyle='--', color='k', lw=1.0,
                   label="Reference")

    ax.plot(xs, ys, color=color, lw=1.0, zorder=1, label=simulator)
    ax.plot(xs, ys - ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    ax.plot(xs, ys + ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    if first:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

for target_file in sys.argv[1:]:
    print "Processing " + target_file
    results = pynam.utils.loadmat(target_file)

    # Iterate over all experiments
    for experiment in results:
        # Skip special keys
        if experiment.startswith("__"):
            continue

        keys = results[experiment]["keys"]
        dims = results[experiment]["dims"]
        data = results[experiment]["data"]
        times = results[experiment]["time"]
        simulator = results[experiment]["simulator"]
        color = 'k'
        if simulator in SIMULATOR_LABELS:
            color = SIMULATOR_COLORS[simulator]
            simulator = SIMULATOR_LABELS[simulator]

        if dims != 1:
            print "Only one-dimensional experiments are supported (yet)"
            print "Skipping experiment " + experiment
            continue

        xlabel = keys[0]
        if xlabel in DIM_LABELS:
            xlabel = DIM_LABELS[xlabel]

        means, stds = calc_means_stds(data, dims)

        # Plot the information metric
        ax, first, _ = get_figure(experiment, "info", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, dims],
                ys_std=stds[:, dims], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="Information $I$ [bit]",
                ys_ref=means[:, dims + 1], first=first)

        # Plot the number of false positives
        ax, first, _ = get_figure(experiment, "fp", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, dims + 2],
                ys_std=stds[:, dims + 2], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="False positives $f_p$ [bit]",
                ys_ref=means[:, dims + 3], first=first)

        # Plot the number of false negatives
        ax, first, _ = get_figure(experiment, "fn", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, dims + 4],
                ys_std=stds[:, dims + 4], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="False negatives $f_n$ [bit]",
                first=first)

        # Plot the latencies
        ax, first, id_ = get_figure(experiment, "latency", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, dims + 5],
                ys_std=stds[:, dims + 5], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="Latency $\\delta$ [ms]",
                first=first)

        # Plot the times
        ax, first, id_ = get_figure(experiment, "times", simulator)
        ax.bar(id_ - 1, times["total"], width=0.35, color="#3465a4")
        ax.bar(id_ - 1, times["sim"], width=0.35, color="#4e9a06")
        if first:
            ax.set_ylabel("Simulation time $t$ [s]")

# Finalize the plots, save them as PDF
for experiment in figures:
    for measure in figures[experiment]:
        fig = figures[experiment][measure]["figure"]
        ax = figures[experiment][measure]["axis"]
        count = figures[experiment][measure]["count"]
        simulators = figures[experiment][measure]["simulators"]

        ax.set_title(experiment)
        if measure == "times":
            ax.set_xticks(np.arange(count) + 0.175)
            ax.set_xticklabels(simulators)
            pTotal = mpatches.Patch(color="#3465a4")
            pSim = mpatches.Patch(color="#4e9a06")
            ax.legend([pTotal, pSim], ["Total", "Simulation Only"],
                    loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=count + 1)
        else:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=count + 1)

        if not os.path.exists("out"):
            os.mkdirs("out")
        fig.savefig("out/plot_" + measure + ".pdf", format='pdf',
                bbox_inches='tight')

