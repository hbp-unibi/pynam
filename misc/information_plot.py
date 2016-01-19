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
    "data.n_bits": "Memory size $n, m$",
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
    "topology.params.v_thresh": "Threshold potential $E_{\\mathrm{Th}}$ [mV]",
    "topology.params.g_leak": "Leak conductivity $g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.params.tau_syn_E": "Excitatory time constant $\\tau_\\mathrm{e}$ [ms]",
    "topology.params.tau_syn_I": "Inhibitory time constant $\\tau_\\mathrm{i}$ [ms]",
    "topology.param_noise.cm": "Membrane capacitance noise $\\sigma C_M$ [nF]",
    "topology.param_noise.e_rev_E": "Excitatory reversal potential noise $\\sigma E_E$ [mV]",
    "topology.param_noise.e_rev_I": "Inhibitory reversal potential noise $\\sigma E_I$ [mV]",
    "topology.param_noise.v_rest": "Resting potential noise $\\sigma E_L$ [mV]",
    "topology.param_noise.v_reset": "Reset potential noise $\\sigma E_{\\mathrm{reset}}$ [mV]",
    "topology.param_noise.v_thresh": "Threshold potential noise $\\sigma E_{\\mathrm{Th}}$ [mV]",
    "topology.param_noise.g_leak": "Leak conductivity noise $\\sigma g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.param_noise.tau_syn_E": "Excitatory time constant noise $\\sigma \\tau_\\mathrm{e}$ [ms]",
    "topology.param_noise.tau_syn_I": "Inhibitory time constant noise $\\sigma \\tau_\\mathrm{i}$ [ms]",
    "topology.multiplicity": "Neuron population size $s$",
    "topology.w": "Synapse weight $w$ [$\\mu\\mathrm{S}$]",
    "topology.sigma_w": "Synapse weight noise $\\sigma_w$ [$\\mu \\mathrm{S}$]",
    "input.burst_size": "Input burst size $s$",
    "input.time_window": "Time window $T$ [ms]",
    "input.isi": "Burst inter-spike-interval $\Delta t$ [ms]",
    "input.sigma_t": "Spike time noise $\sigma_t$ [ms]",
    "input.sigma_t_offs": "Spike time offset noise $\sigma_t^{\\mathrm{offs}}$ [ms]",
}

DIM_LABELS_SMALL = {
    "data.n_bits": "$n, m$",
    "data.n_bits_in": "$m$",
    "data.n_bits_out": "$n$",
    "data.n_ones_in": "$c$",
    "data.n_ones_out": "$d$",
    "data.n_samples": "$N$",
    "topology.params.cm": "$C_M$ [nF]",
    "topology.params.e_rev_E": "$E_E$ [mV]",
    "topology.params.e_rev_I": "$E_I$ [mV]",
    "topology.params.v_rest": "$E_L$ [mV]",
    "topology.params.v_reset": "$E_{\\mathrm{reset}}$ [mV]",
    "topology.params.v_thresh": "$E_{\\mathrm{Th}}$ [mV]",
    "topology.params.g_leak": "$g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.params.tau_syn_E": "$\\tau_\\mathrm{e}$ [ms]",
    "topology.params.tau_syn_I": "$\\tau_\\mathrm{i}$ [ms]",
    "topology.param_noise.cm": "$\\sigma C_M$ [nF]",
    "topology.param_noise.e_rev_E": "$\\sigma E_E$ [mV]",
    "topology.param_noise.e_rev_I": "$\\sigma E_I$ [mV]",
    "topology.param_noise.v_rest": "$\\sigma E_L$ [mV]",
    "topology.param_noise.v_reset": "$\\sigma E_{\\mathrm{reset}}$ [mV]",
    "topology.param_noise.v_thresh": "$\\sigma E_{\\mathrm{Th}}$ [mV]",
    "topology.param_noise.g_leak": "$\\sigma g_\\mathrm{L}$ [$\\mu\\mathrm{S}$]",
    "topology.param_noise.tau_syn_E": "$\\sigma \\tau_\\mathrm{e}$ [ms]",
    "topology.param_noise.tau_syn_I": "$\\sigma \\tau_\\mathrm{i}$ [ms]",
    "topology.multiplicity": "$s$",
    "topology.w": "$w$ [$\\mu\\mathrm{S}$]",
    "topology.sigma_w": "$\\sigma_w$ [$\\mu \\mathrm{S}$]",
    "input.burst_size": "$s$",
    "input.time_window": "$T$ [ms]",
    "input.isi": "$\Delta t$ [ms]",
    "input.sigma_t": "$\sigma_t$ [ms]",
    "input.sigma_t_offs": "$\sigma_t^{\\mathrm{offs}}$ [ms]",
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

SMALL = False

figures = {}

def cm2inch(value):
    return value / 2.54

def get_figure(experiment, measure, simulator, figsize=None, bottomax=False):
    global figures
    first = False
    if not experiment in figures:
        figures[experiment] = {}
    if not measure in figures[experiment]:
        first = True
        figures[experiment][measure] = {}
        if figsize is None:
            if "info" in measure:
                figsize = (cm2inch(12.8), cm2inch(5.0))
            else:
                figsize = (cm2inch(5.5), cm2inch(5.0))
        fig = plt.figure(figsize=figsize)
        if not bottomax or SMALL:
            ax = fig.add_subplot(111)
        else:
            ax1 = fig.add_axes([0.0, 0.15, 1.0, 0.85])
            ax2 = fig.add_axes([0.0, 0.0, 1.0, 0.05])
            ax = (ax1, ax2)
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
        ys_ref=None, first=True, ymin=None, ymax=None):
    if first:
        if not ys_ref is None:
            ax.plot(xs, ys_ref, linestyle='--', color='k', lw=1.0)
#                   label="Reference")

    ax.plot(xs, ys, color=color, lw=1.0, zorder=1, label=simulator)
    ax.plot(xs, ys - ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    ax.plot(xs, ys + ys_std * 0.5, lw=0.5, linestyle=':', color=color, zorder=0)
    ax.set_xlim(np.min(xs), np.max(xs))
    if first:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if (ymin is None) and (ymax is None):
            ax.autoscale_view()
        elif ymax is None:
            ax.set_ylim(bottom=ymin)
        elif ymin is None:
            ax.set_ylim(top=ymax)
        else:
            ax.set_ylim(bottom=ymin,top=ymax)

def plot_measure2d(ax, xs, ys, zs, simulator, xlabel, ylabel, zlabel="", vmin=None,
        vmax=None, qualitative=False):

    midx = lambda m: keys.index(m)

    if SMALL:
        ax1 = ax
    else:
        ax1 = ax[0]
        ax2 = ax[1]

    _, steps_x = np.unique(xs, return_counts=True)
    _, steps_y = np.unique(ys, return_counts=True)
    steps_x = np.max(steps_x)
    steps_y = np.max(steps_y)
    xs = xs.reshape((steps_x, steps_y))
    ys = ys.reshape((steps_x, steps_y))
    zs = zs.reshape((steps_x, steps_y))

    # Select the colormap
    if qualitative:
        cmap = "rainbow"
    else:
        if "latency" in zlabel:
            cmap = "Greens"
        else:
            cmap = "Purples"
        if vmin < 0.0:
            cmap = "PuOr"

    # Auto-scale
    idcs = zs != np.inf
    if np.sum(idcs) == 0:
        return
    if vmin is None:
        vmin = np.min(zs[idcs])
    if vmax is None:
        vmax = np.max(zs[idcs])

    extent = (np.min(xs), np.max(xs), np.min(ys), np.max(ys))
    ax1.imshow(zs, aspect='auto', origin='lower', extent=extent, cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="none")

    levels = np.linspace(vmin, vmax, 11)
    CS2 = ax1.contour(xs, ys, zs, levels, linewidths=0.25, colors='k',
            vmin=vmin, vmax=vmax)
    ax1.grid()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if not SMALL:
        cbar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                orientation='horizontal', ticks=levels,
                norm=matplotlib.colors.Normalize(vmin, vmax))
        cbar.set_label(zlabel)


def get_label(key):
    if SMALL:
        return DIM_LABELS_SMALL[key] if key in DIM_LABELS_SMALL else key
    else:
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
                xlabel=xlabel, ylabel="Information $I$ [bit]",
                ymin=0,
                ys_ref=means[:, midx("I_ref")], first=first)

    # Plot the information metric (normalized)
    if "I_n" in keys:
        ax, first, _ = get_figure(experiment, "info_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("I_n")] * 100.0,
                ys_std=stds[:, midx("I_n")] * 100.0, color=color,
                simulator=simulator,
                ymin=0, ymax=100,
                xlabel=xlabel, ylabel="Information $I$ [\\%]", first=first)

    # Plot the number of false positives
    if "fp" in keys:
        ax, first, _ = get_figure(experiment, "fp", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fp")],
                ys_std=stds[:, midx("fp")], color=color, simulator=simulator,
                ymin=0, ymax=300,
                xlabel=xlabel, ylabel="False positives $n_\\mathrm{fp}$ [bit]",
                ys_ref=means[:, midx("fp_ref")], first=first)

    # Plot the number of false positives (normalized)
    if "fp_n" in keys:
        ax, first, _ = get_figure(experiment, "fp_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fp_n")] * 100.0,
                ys_std=stds[:, midx("fp_n")] * 100.0, color=color,
                simulator=simulator,
                ymin=0,
                xlabel=xlabel, ylabel="False positives $n_\\mathrm{fp}$ [\\%]",
                ys_ref=means[:, midx("fp_ref_n")] * 100.0, first=first)

    # Plot the number of false negatives
    if "fn" in keys:
        ax, first, _ = get_figure(experiment, "fn", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fn")],
                ys_std=stds[:, midx("fn")], color=color, simulator=simulator,
                xlabel=xlabel, ylabel="False negatives $n_\\mathrm{fn}$ [bit]",
                ymin=0,
                first=first)

    # Plot the number of false negatives (normalized)
    if "fn_n" in keys:
        ax, first, _ = get_figure(experiment, "fn_n", simulator)
        plot_measure(ax, xs=means[:, 0], ys=means[:, midx("fn_n")] * 100.0,
                ys_std=stds[:, midx("fn_n")] * 100.0, color=color,
                simulator=simulator, xlabel=xlabel,
                ymin=0, ymax=100,
                ylabel="False negatives $n_\\mathrm{fn}$ [\\%]", first=first)

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
    if SMALL:
        figsize = (cm2inch(5.6), cm2inch(5.6))
    else:
        figsize = (cm2inch(10), cm2inch(9.5))

    # Plot the information metric
    if "I_n" in keys:
        ax, _, _ = get_figure(experiment, "info_n" + "_2d_" + simulator,
                simulator, figsize=figsize, bottomax=(not SMALL))
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("I_n")], simulator=simulator,
                qualitative=True,
                zlabel="Realative information $I$",
                xlabel=xlabel, ylabel=ylabel, vmin=0.0, vmax=1.0)

    # Plot the false positives metric
    if ("fp_n" in keys) and ("fp_ref_n" in keys):
        ax, _, _ = get_figure(experiment, "fp_n" + "_2d_" + simulator,
                simulator, figsize=figsize, bottomax=True)
        zs_ref = means[:, midx("fp_ref_n")]
        zs = means[:, midx("fp_n")] - zs_ref
        zs[zs > 0] = zs[zs > 0] / (1 - zs_ref[zs > 0])
        zs[zs < 0] = zs[zs < 0] / zs_ref[zs < 0]
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=zs,
                simulator=simulator, xlabel=xlabel, ylabel=ylabel,
                zlabel="Normalised false positive count",
                vmin=-1.0, vmax=1.0)

    # Plot the false negatives metric
    if "fn_n" in keys:
        ax, _, _ = get_figure(experiment, "fn_n" + "_2d_" + simulator,
                simulator, figsize=figsize, bottomax=True)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("fn_n")],
                simulator=simulator, xlabel=xlabel, ylabel=ylabel,
                zlabel="Normalised false negative count",
                vmin=0.0, vmax=1.0)

    # Plot the latency metric
    if "lat_avg" in keys:
        ax, _, _ = get_figure(experiment, "lat_avg" + "_2d_" + simulator,
                simulator, figsize=figsize, bottomax=True)
        plot_measure2d(ax, xs=means[:, 0], ys=means[:, 1],
                zs=means[:, midx("lat_avg")],
                simulator=simulator, xlabel=xlabel, ylabel=ylabel,
                zlabel="Average latency $\delta$ [ms]",
                vmin=0.0, vmax=50.0)

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

        if isinstance(ax, tuple):
            ax = ax[0]
#        if (not SMALL):
#            if (len(simulators) == 1) and (simulators[0] in SIMULATOR_LABELS):
#                ax.set_title(SIMULATOR_LABELS[simulators[0]] + " " + experiment)
#            else:
#                ax.set_title(experiment)
        if measure == "times":
            ax.set_xticks(np.arange(count) + 0.175)
            ax.set_xticklabels(simulators)
            pTotal = mpatches.Patch(color="#3465a4")
            pSim = mpatches.Patch(color="#4e9a06")
            ax.legend([pTotal, pSim], ["Total", "Simulation Only"],
                    loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=4)
        elif "info" in measure:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
                    ncol=4)

        if not os.path.exists("out"):
            os.mkdirs("out")
        if SMALL:
            fig.savefig("out/plot_" + str(i) + "_" + measure + "_small.pdf", format='pdf',
                    bbox_inches='tight')
        else:
            fig.savefig("out/plot_" + str(i) + "_" + measure + ".pdf", format='pdf',
                    bbox_inches='tight')

