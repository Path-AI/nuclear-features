# core python
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import List

# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# survival analysis functions
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from plotly.offline import plot

# significance tests
from scipy import stats

# dendrogram plots
from scipy.cluster import hierarchy as sch

from .utils import robust_z


def pathai_clustermap(
    cluster,
    row_cluster_name=None,
    col_cluster_name=None,
    label_df=None,
    label_key=None,
    norm_viz=None,
    norm_axis=0,
    hx=None,
    ax=None,
    bx=None,
    lx=None,
    fig=None,
    cbarx=None,
    figsize=(3.5, 2.5),
    show_xticks=True,
    label_cmap=None,
    **kwargs,
):
    """Replaces seaborn clustermap visualization for clustering we have completed on the cluster object. Note that
    teh zscoring argument here only applies for the visualization and care should be taken to ensure the
    visualization is consistent with the clustering. Row and column cluster names refer to keys to the dictionary of
    clustering results. Note that no actual clustering is done within this function, it is only for plotting..
    """

    # get cluster dict
    cd = cluster.cluster

    # set up what we will be returning
    returns = {}

    # set up the plot if one is not provided
    if ax is None:

        num_cols = 2
        num_rows = 1
        width_ratios = [10, 1]
        height_ratios = [10]
        if type(label_df) == pd.DataFrame:
            width_ratios = [0.5] + width_ratios
            num_cols += 1

        if row_cluster_name:
            num_cols += 1
            width_ratios = [2] + width_ratios

        if col_cluster_name:
            num_rows += 1
            height_ratios = [2] + height_ratios

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            num_rows,
            num_cols,  # wspace=.01, hspace=.01, left=.01, right=.99, bottom=.01, top=.99,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        cbarx = plt.subplot(gs[-1, -1])
        hx = plt.subplot(gs[-1, -2])

        if row_cluster_name:
            ax = plt.subplot(gs[-1, 0])
            ax.set_axis_off()

        if col_cluster_name:
            bx = plt.subplot(gs[0, -2])
            bx.set_axis_off()

        if type(label_df) == pd.DataFrame:
            lx = plt.subplot(gs[-1, 1])
            lx.set_axis_off()

    # get dendrograms
    dendr = dendc = None
    if row_cluster_name:
        dendr = _plot_dendrogram_component(ax, cd[row_cluster_name], orientation="left")
        returns["row_cluster_labels"] = cd[row_cluster_name]["clust"][dendr["leaves"]]
        returns["row_cluster_names"] = np.array(cluster.df.index)[dendr["leaves"]]
    if col_cluster_name:
        dendc = _plot_dendrogram_component(bx, cd[col_cluster_name], orientation="top")
        returns["col_cluster_labels"] = cd[col_cluster_name]["clust"][dendc["leaves"]]
        returns["col_cluster_names"] = np.array(cluster.df.columns)[dendr["leaves"]]

    # get the pcolormesh plot set up
    # z-score if requested
    df = cluster.df.copy()
    assert norm_viz in ["z", "robust_z", None]
    if norm_viz == "z":
        df = df.apply(stats.zscore, axis=norm_axis, ddof=1)
    elif norm_viz == "robust_z":
        df = robust_z(df, axis=norm_axis)

    # get values, sort by how it is clustered
    plot_data = df.values
    if dendc:
        assert (
            len(dendc["leaves"]) == plot_data.shape[1]
        ), "Mismatch between cluster and column shape."
        plot_data = plot_data[:, dendc["leaves"]]
    if dendr:
        assert (
            len(dendr["leaves"]) == plot_data.shape[0]
        ), "Mismatch between cluster and row shape."
        plot_data = plot_data[dendr["leaves"]]
    mesh = hx.pcolormesh(plot_data, **kwargs)

    hx.set_yticklabels([])
    if show_xticks:
        xlabels = df.keys()
        if col_cluster_name:
            xlabels = xlabels[:, dendc["leaves"]]
        hx.set_xticks(np.arange(len(xlabels)) + 0.5)
        hx.set_xticklabels(xlabels)
        plt.setp(hx.xaxis.get_majorticklabels(), rotation=90)

    cbar = plt.colorbar(mesh, cax=cbarx)
    cbar.set_label(cluster.general_properties["name"])
    hx.invert_yaxis()
    if ax is not None:
        ax.invert_yaxis()
    returns["fig"] = fig
    returns["axes"] = [hx, cbarx, ax, bx]

    # plotting of the labels as qualitative values results in returning also labels and their associated colors
    if type(label_df) == pd.DataFrame:
        # if labels exist
        assert all(df.index == label_df.index), "Labeling index must match data index."
        plot_ldf = pd.DataFrame(label_df[label_key])
        plot_ldf.fillna("nan", inplace=True)
        label_data = label_df[label_key].values.astype(str)
        labels = np.unique(np.array(label_data, dtype=str))
        if label_cmap:
            label_map = label_cmap  # we're trusting the user here
        else:
            label_map = ListedColormap(
                ["blue", "red", "yellow", "black", "white", "purple"][: len(labels)]
            )
        for i, lb in enumerate(labels):
            plot_ldf.replace(lb, i, inplace=True)

        plot_lvalues = np.array(plot_ldf.values)
        if dendr:
            plot_lvalues = plot_lvalues[dendr["leaves"]]
        lx.pcolormesh(plot_lvalues, cmap=label_map)
        lx.invert_yaxis()
        returns["labels"] = labels
        returns["label_colors"] = label_map.colors
        returns["axes"] = [hx, cbarx, ax, bx, lx]

    return returns


def _plot_dendrogram_component(ax, cluster_dict, **kwargs):
    """Helper to plot the"""
    dend = sch.dendrogram(
        cluster_dict["link"], color_threshold=cluster_dict["thresh"], ax=ax, **kwargs
    )
    return dend


def qq_plot_pvals(pvals, rejected_h0=None, **kwargs):

    """
    quantile-quantile plot showing distribution of observed pvalues against expected given the null distribution
    :param pvals: P values
    :param rejected_h0: if desired indices of rejected null hypotheses are highlighted in red
    """
    fig, ax = plt.subplots()
    pvals = np.array(pvals)
    n_unif = np.sort(-np.log10(np.linspace(1.0 / len(pvals), 1, len(pvals))))
    plt.plot(np.sort(n_unif), np.sort(-np.log10(np.array(pvals))), ".", **kwargs)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Expected p-values (-log$_{10}$(p))")
    ax.set_ylabel("Observed p-values (-log$_{10}$(p))")
    if rejected_h0 is not None:
        sig_ps = pvals[rejected_h0]
        sig_ps.sort()
        sig_ps = -np.log10(sig_ps[::-1])
        quantiles = np.sort(n_unif[-len(sig_ps) :])
        ax.scatter(quantiles, sig_ps, facecolors="none", edgecolors="r")
    return fig, ax


# in development:
def dev_plot_lr(exog_key, endog_key, data_df, models_df, model_dict=None, ax=None, fig=None):
    exog_vals = data_df[exog_key].values
    endog_vals = data_df[endog_key].values
    model = models_df.loc[exog_key]

    if not ax:
        plt.figure()
        ax = plt.subplot()

    ax.scatter(exog_vals, endog_vals, color="k")
    xvals = np.linspace(0, np.max(exog_vals), 1000)
    yvals = 1 / (1 + np.exp(-1 * (model["const"] + model["beta"] * xvals)))
    ax.plot(xvals, yvals, color="red", lw=1.0)
    ax.set_xlabel(exog_key)
    ax.set_ylabel(f"P({endog_key})")

    if model_dict:
        lr_object = model_dict[exog_key]
        params = lr_object.params
        conf = lr_object.conf_int()
        conf["Odds Ratio"] = params
        conf.columns = ["5%", "95%", "Odds Ratio"]
        if len(exog_key) > 15:
            as_list = conf.index.tolist()
            as_list[1] = "feature"
            conf.index = as_list
        print(np.exp(conf))

# padding for formatting figures using tight_layout
layout_pad = {"pad": 0.05, "h_pad": 0.6, "w_pad": 0.6}

def _is_member(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, np.nan) for itm in a]
