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
import scipy.io as sio

# Include the PyNAM folder
import sys
import os
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), "../pynam"))

import binam
import binam_data
import binam_utils

colors = [
    '#800000', # 0
    '#47294F', # 1
    '#193A6B', # 2
    '#A08300', # 3
    '#AA4C00', # 4
    '#193A6B', # 5
    '#800000', # 6
    '#AA4C00', # 7
    '#3C7705', # 9
    '#800000', # 10
    '#47294F', # 11
    '#3C7705', # 12
    '#A08300', # 13
]

title_rewrites = {
    "Random":
        "Random, with duplicates",
    "x: Random, no duplicates; y: Random, with duplicates":
        "$\\vec x$: Random; $\\vec y$: Random",
    "x: Balanced, no duplicates; y: Random, with duplicates":
        "$\\vec x$: Balanced; $\\vec y$: Random",
    "x: Balanced, no duplicates; y: Balanced, with duplicates":
        "$\\vec x$: Balanced; $\\vec y$: Balanced",
}

def cm2inch(value):
    return value / 2.54

def reorder_legend(elems, labels, ncols=2):
    nelems = len(elems)
    clen = int(math.ceil(float(nelems) / float(ncols)))

    elems_res = [None for _ in xrange(nelems)]
    labels_res = [None for _ in xrange(nelems)]
    for i in xrange(nelems):
        j = i // clen + (i % clen) * ncols
        elems_res[i] = elems[j]
        labels_res[i] = labels[j]
    return elems_res, labels_res

def plot(results, datasets):
    fig = plt.figure(figsize=(cm2inch(11.8), cm2inch(8)))
    ax = fig.add_subplot(111)
    t = results["t"]
    if len(datasets) == 0:
        datasets = range(len(results["info"]))
    legend_elems = []
    legend_labels = []
    for i in datasets:
        data = results["info"][i]
        title = results["title"][i].strip()
        if title in title_rewrites:
            title = title_rewrites[title]
        color = colors[i % len(colors)]
        i25, i75 = np.percentile(data, [25, 75], 0)
        ax.plot(t, np.median(data, 0), lw=1, color=color, zorder=1)
        ax.plot(t, data[np.argmax(np.max(data, 1))], ':', lw=0.5, color=color,
                zorder=2)
        ax.fill_between(t, i25, i75, facecolor=color, alpha=0.3, zorder=0, lw=0)
        ax.fill_between(t, np.min(data, 0), np.max(data, 0), facecolor=color,
                alpha=0.15, zorder=-1, lw=0)
        legend_elems.append(mlines.Line2D([], [], color=color, lw=1))
        legend_labels.append(title)

    refInfo = np.zeros(len(t))
    for i in xrange(1, len(t)):
        refInfo[i] = binam_utils.expected_entropy(t[i], results["n_bits"],
            results["n_ones"])
    plt.plot(t, refInfo, '--', color='k', lw=0.5, zorder=2)

    color='#2e3436'
    p1 = mlines.Line2D([], [], linestyle=':', color=color, lw=0.5)
    p2 = mlines.Line2D([], [], linestyle='--', color='k', lw=0.5)
    p3 = mpatches.Patch(color=color, alpha=0.3)
    p4 = mpatches.Patch(color=color, alpha=0.15)
    legend_elems = legend_elems + [p1, p2, p3, p4]
    legend_labels = legend_labels + ["Trial with max. information",
        "Expected information", "$25/75\%$-quantile", "Min./Max."]
    legend_elems, legend_labels = reorder_legend(legend_elems, legend_labels)

    plt.legend(legend_elems, legend_labels, loc='lower center',
        bbox_to_anchor=(0.5, 1.05), ncol=2)

    ax.set_xlabel('Number of trained samples $N$')
    ax.set_ylabel('Information $I$ [Bit]')
    ax.set_xlim(np.min(t), np.max(t))
    plt.savefig(sys.argv[1] + ".pdf", format='pdf', bbox_inches='tight')

if (len(sys.argv) < 2):
    print "Usage: ./binam_data_information_plot.py <FILE> [<DATASETS>]"
    sys.exit(1)

results = sio.loadmat(sys.argv[1], squeeze_me=True, struct_as_record=False)
datasets = [int(s) for s in sys.argv[2:]]
plot(results, datasets)
plt.show()
