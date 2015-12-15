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
import math
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
    "topology.params.g_leak": "Leak conductivity $g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.params.tau_syn_E": "Excitatory time constant $\\tau_\\mathrm{e}$ [ms]",
    "topology.params.tau_syn_I": "Inhibitory time constant $\\tau_\\mathrm{i}$ [ms]",
    "topology.param_noise.cm": "Membrane capacitance noise $\\sigma C_M$ [nF]",
    "topology.param_noise.e_rev_E": "Excitatory reversal potential noise $\\sigma E_E$ [mV]",
    "topology.param_noise.e_rev_I": "Inhibitory reversal potential noise $\\sigma E_I$ [mV]",
    "topology.param_noise.v_rest": "Resting potential noise $\\sigma E_L$ [mV]",
    "topology.param_noise.v_reset": "Reset potential noise $\\sigma E_{\\mathrm{reset}}$ [mV]",
    "topology.param_noise.v_thresh": "Threshold potential noise $\\sigma E_{\\mathrm{thresh}}$ [mV]",
    "topology.param_noise.g_leak": "Leak conductivity noise $\\sigma g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.param_noise.tau_syn_E": "Excitatory time constant noise $\\sigma \\tau_\\mathrm{e}$ [ms]",
    "topology.param_noise.tau_syn_I": "Inhibitory time constant noise $\\sigma \\tau_\\mathrm{i}$ [ms]",
    "topology.multiplicity": "Neuron population size $s$",
    "topology.w": "Synapse weight $w$ [$\\mu\\mathrm{S}$]",
    "topology.sigma_w": "Synapse weight noise $\\sigma w$ [$\\mu \\mathrm{S}$]",
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
    "spiNNaker": "NM-MC1",
    "spikey": "Spikey",
    "nest": "NEST"
}

# Colors for all simulators
SIMULATOR_COLORS = {
    "ess": '#73d216',
    "nmpm1": "#75507b",
    "spiNNaker": "#f57900",
    "spikey": "#cc0000",
    "nest": "#3465a4"
}

figures = {}

def cm2inch(value):
    return value / 2.54

def get_figure(experiment, measure, simulator, figsize=None):
    global figures
    first = False
    if not experiment in figures:
        figures[experiment] = {}
    if not measure in figures[experiment]:
        first = True
        figures[experiment][measure] = {}
        if figsize is None:
            figsize = (cm2inch(8), cm2inch(6))
        fig = plt.figure(figsize=figsize)
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
        while end < data.shape[0] and np.all(values == data[end, 0:dims]):
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
            ax.plot(xs, ys_ref, linestyle='--', color='k', lw=1.0)
#                   label="Reference")

    ax.plot(xs, ys, color=color, lw=1.0, zorder=1, label=simulator)
    ax.plot(xs, ys - ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    ax.plot(xs, ys + ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    if first:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

def plot_measure2d(ax, xs, ys, zs, simulator, xlabel, ylabel, vmin=None,
        vmax=None, fmt='%.0f\\%%'):

    midx = lambda m: keys.index(m)

    _, steps_x = np.unique(xs, return_counts=True)
    _, steps_y = np.unique(ys, return_counts=True)
    steps_x = np.max(steps_x)
    steps_y = np.max(steps_y)
    xs = xs.reshape((steps_x, steps_y))
    ys = ys.reshape((steps_x, steps_y))
    zs = zs.reshape((steps_x, steps_y))

    # Auto-scale
    idcs = zs != np.inf
    if np.sum(idcs) == 0:
        return
    if vmin is None:
        vmin = np.min(zs[idcs])
    if vmax is None:
        vmax = np.max(zs[idcs])

    # Select the colormap
    cmap = "Purples"
    if vmin < 0.0:
        cmap = "PuOr"

    steps = math.pow(10.0, round(math.log10(vmax - vmin)) - 1.0)
    levels = np.arange(vmin, vmax, steps)
    CS1 = ax.contourf(xs, ys, zs, levels, cmap=cmap, vmin=vmin, vmax=vmax)
    CS2 = ax.contour(xs, ys, zs, levels, colors='k')
    CS2.levels = map(lambda val: fmt % val, levels)
    ax.clabel(CS2, inline=1, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def get_label(key):
    return DIM_LABELS[key] if key in DIM_LABELS else key

def plot_1d(xlabel, means, stds, simulator, keys):
    dims = 1

    midx = lambda m: keys.index(m)

    color = 'k'
    if simulator in SIMULATOR_LABELS:
        color = SIMULATOR_COLORS[simulator]
        simulator = SIMULATOR_LABELS[simulator]

    # Plot the information metric
    if "I" in keys:
        ax, first, _ = get_figure(experiment, "info", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("I")],
                ys_std=stds[:, midx("I")], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="Storage capacity $S$ [bit]",
                ys_ref=means[:, midx("I_ref")], first=first)

    # Plot the information metric (normalized)
    if "I_n" in keys:
        ax, first, _ = get_figure(experiment, "info_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("I_n")] * 100.0,
                ys_std=stds[:, midx("I_n")] * 100.0, color=color,
                simulator=simulator,
                xlabel=xlabel, ylabel="Storage capacity $S$ [\\%]", first=first)

    # Plot the number of false positives
    if "fp" in keys:
        ax, first, _ = get_figure(experiment, "fp", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fp")],
                ys_std=stds[:, midx("fp")], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="False positives $f_p$ [bit]",
                ys_ref=means[:, midx("fp_ref")], first=first)

    # Plot the number of false positives (normalized)
    if "fp_n" in keys:
        ax, first, _ = get_figure(experiment, "fp_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fp_n")] * 100.0,
                ys_std=stds[:, midx("fp_n")] * 100.0, color=color,
                simulator=simulator,
                xlabel=xlabel, ylabel="False positives $f_p$ [\\%]",
                ys_ref=means[:, midx("fp_ref_n")] * 100.0, first=first)

    # Plot the number of false negatives
    if "fn" in keys:
        ax, first, _ = get_figure(experiment, "fn", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fn")],
                ys_std=stds[:, midx("fn")], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="False negatives $f_n$ [bit]",
                first=first)

    # Plot the number of false negatives (normalized)
    if "fn_n" in keys:
        ax, first, _ = get_figure(experiment, "fn_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fn_n")] * 100.0,
                ys_std=stds[:, midx("fn_n")] * 100.0, color=color,
                simulator=simulator, xlabel=xlabel,
                ylabel="False negatives $f_n$ [\\%]", first=first)

    # Plot the latencies
    if "lat_avg" in keys:
        ax, first, id_ = get_figure(experiment, "latency", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("lat_avg")],
                ys_std=stds[:, midx("lat_avg")], color=color,
                simulator=simulator, xlabel=xlabel,
                ylabel="Latency $\\delta$ [ms]", first=first)


def plot_2d(xlabel, ylabel, means, stds, simulator, keys):
    dims = 2

    midx = lambda m: keys.index(m)
    figsize = (cm2inch(10), cm2inch(10))

    # Plot the information metric
    if "I_n" in keys:
        ax, _, _ = get_figure(experiment, "info_n" + "_2d_" + simulator,
                simulator, figsize=figsize)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("I_n")] * 100.0, simulator=simulator,
                xlabel=xlabel, ylabel=ylabel, vmin=0.0, vmax=100.0)

    # Plot the false positives metric
    if ("fp_n" in keys) and ("fp_ref_n" in keys):
        ax, _, _ = get_figure(experiment, "fp_n" + "_2d_" + simulator,
                simulator, figsize=figsize)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=(means[:, midx("fp_n")] - means[:, midx("fp_ref_n")]) * 100,
                simulator=simulator, xlabel=xlabel, ylabel=ylabel,
                vmin=-100.0, vmax=100.0)

    # Plot the false negatives metric
    if "fn_n" in keys:
        ax, _, _ = get_figure(experiment, "fn_n" + "_2d_" + simulator,
                simulator, figsize=figsize)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("fn_n")] * 100.0,
                simulator=simulator, xlabel=xlabel, ylabel=ylabel,
                vmin=0.0, vmax=100.0)

    # Plot the latency metric
    if "lat_avg" in keys:
        ax, _, _ = get_figure(experiment, "lat_avg" + "_2d_" + simulator,
                simulator, figsize=figsize)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("lat_avg")],
                simulator=simulator, xlabel=xlabel, ylabel=ylabel, vmin=0.0,
                fmt="%.2f")

#
# Main entry point
#

for target_file in sys.argv[1:]:
    print "Processing " + target_file
    results = pynam.utils.loadmat(target_file)

    # Iterate over all experiments
    for experiment in results:
        # Skip special keys
        if experiment.startswith("__"):
            continue

        keys = map(lambda s: s.strip(), results[experiment]["keys"].tolist())
        dims = results[experiment]["dims"]
        data = results[experiment]["data"]
        times = results[experiment]["time"]
        simulator = results[experiment]["simulator"]

        means, stds = calc_means_stds(data, dims)

        if dims == 1:
            plot_1d(xlabel=get_label(keys[0]), means=means, stds=stds,
                    simulator=simulator, keys=keys)
        elif dims == 2:
            plot_2d(xlabel=get_label(keys[0]), ylabel=get_label(keys[1]),
                    means=means, stds=stds, simulator=simulator, keys=keys)
        else:
            print "Only one and two-dimensional experiments are supported (yet)"
            print "Skipping experiment " + experiment
            continue

        # Plot the times
        if simulator != "ESS":
            ax, first, id_ = get_figure(experiment, "times", simulator)
            ax.bar(id_ - 1, times["total"], width=0.35, color="#3465a4")
            ax.bar(id_ - 1, times["sim"], width=0.35, color="#4e9a06")
            if first:
                ax.set_ylabel("Simulation time $t$ [s]")

# Finalize the plots, save them as PDF
for i, experiment in enumerate(figures):
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
                    ncol=2)
        else:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=2)

        if not os.path.exists("out"):
            os.mkdirs("out")
        fig.savefig("out/plot_" + str(i) + "_" + measure + ".pdf", format='pdf',
                bbox_inches='tight')

