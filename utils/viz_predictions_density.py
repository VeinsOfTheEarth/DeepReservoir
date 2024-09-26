#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Functionality for plotting predicted vs. ground truth values
   (e.g. to compare the ouput of a ML-based regression model).
"""

from __future__ import print_function

import numpy as np

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

np.set_printoptions(precision=2)

def plot_true_vs_predicted_density(y_test, y_pred, xbins=51):
    """Compute density of predicted vs. ground truth.

    A 2D histrogram of ground truth vs. predicted is computed. The
    line x=y is also plotted. Ideally when predictions match ground
    truth, the highest density should be registered around the x=y line.
    The x-axis of the resulting plot will correspond to the observed (i.e.
    ground truth values) while the y-axis to the predicted values.

    Arguments
    ---------
    y_test: 1D np.array with ground truth.
    y_pred: 1D np.array with corresponding predictions.
    x_bins: Number of bins per axis to compute the 2D histogram.
    """
    fig = plt.figure(figsize=(11,8))
    ax = plt.gca()
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4.)
    plt.hist2d(y_test, y_pred, bins=xbins, norm=LogNorm())
    cb = plt.colorbar()
    ax.set_xlabel("Observed", fontsize=18, labelpad=15.)
    ax.set_ylabel("Predicted", fontsize=18, labelpad=15.)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.grid()
    plt.title("Observed vs. Predicted", fontsize=20)
    plt.tight_layout()
    plt.savefig("prediction_vs_gtruth_density.png")
    plt.close()
