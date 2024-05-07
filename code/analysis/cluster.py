# coding=utf-8
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy as sch
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

# relative imports
from .abstractions import Featureset
from .utils import robust_z
from .visualizations import layout_pad


class Cluster(Featureset):
    """Class containing methods for clustering analyses of data. Largely based on interfaces to scipy.cluster, with
    additional functionality, plotting tools, and statistical tests.
    """

    def __init__(self, name: str) -> None:
        """Initiatilization of clustering, only takes a name. Initializes a properties dict for tracking what the
        clustering corresponds to."""
        super().__init__(name)
        self.cluster = {}

    def perform_clustering(
        self,
        name: str,
        distance: str = "euclidean",
        linkage: str = "ward",
        threshold: float = 0.7,
        threshold_type: str = "fractional",
        norm: str = "z",
        norm_axis: int = 0,
        cluster_by: str = "rows",
    ) -> None:
        """Performs the clustering according to given parameters. Parameter
        `distance` may also be a function that computes the distance metric for
         two vectors. norm_axis in 0 (cols), 1 (rows)"""

        assert threshold_type in [
            "fractional",
            "absolute",
        ], "Threshold type must be either fractional or absolute."

        # record the properties as it is called
        properties = self.general_properties.copy()
        properties["distance"] = distance
        properties["linkage"] = linkage
        properties["threshold"] = threshold
        properties["threshold_type"] = threshold_type
        properties["clustering_norm"] = f"{norm}, axis={norm_axis}"

        df = self.df.copy()

        # z-score if requested
        assert norm in ["z", "robust_z", None]
        if norm == "z":
            df = df.apply(stats.zscore, axis=norm_axis, ddof=1)
        elif norm == "robust_z":
            df = robust_z(df, axis=norm_axis)

        # perform clustering
        assert cluster_by in ["cols", "rows"], "Clustering must be done by rows or columns."
        if cluster_by == "cols":
            dist = sch.distance.pdist(df.values.T, distance)
        elif cluster_by == "rows":
            dist = sch.distance.pdist(df.values, distance)
        link = sch.linkage(dist, method=linkage)

        if threshold_type == "fractional":
            dist_cutoff = threshold * np.max(link[:, 2])
        elif threshold_type == "absolute":
            dist_cutoff = threshold
        else:
            dist_cutoff = np.max(link)

        clust = sch.fcluster(link, t=dist_cutoff, criterion="distance") - 1

        if cluster_by == "rows":
            cluster_df = pd.DataFrame.from_dict({df.index.name: df.index, "cluster_id": clust})
            cluster_df.set_index(df.index.name, inplace=True)
        elif cluster_by == "cols":
            cluster_df = pd.DataFrame.from_dict(
                {"Feature": df.columns.to_numpy(), "cluster_id": clust}
            )
            cluster_df.set_index("Feature", inplace=True)
        self.cluster[name] = {
            "link": link,
            "dist": dist,
            "clust": clust,
            "thresh": dist_cutoff,
            "properties": properties,
            "df": cluster_df,
        }

    def plot_dendrogram(self, cluster_name: str, ax=None, fig=None, **kwargs):
        """Creates a plot of a dendogram."""
        if ax is None:
            fig = plt.figure(figsize=(3.5, 3))
            ax = plt.subplot()

        cd = self.cluster[cluster_name]
        dend = sch.dendrogram(cd["link"], color_threshold=cd["thresh"], ax=ax, **kwargs)

        ax.set_ylabel("Distance Value")
        plt.tight_layout(**layout_pad)
        return ax, fig, dend

    def plot_cluster_count(self, name, ax=None, fig=None, use_kneefinder: bool = False, **kwargs):
        """Plots how the cluster count varies with threshold."""

        cluster = self.cluster[name]
        # flips through prospective cutoffs and returns how many clusters there are
        cuts = np.linspace(0.02, 1.0, 1000) * np.max(cluster["link"][:, 2])
        clusters_n = np.zeros_like(cuts)
        for idx, c in enumerate(cuts):
            ind = sch.fcluster(cluster["link"], t=c, criterion="distance")
            clusters_n[idx] = np.max(ind)

        if ax is None:
            fig = plt.figure(figsize=(3.5, 3))
            ax = plt.subplot()

        if use_kneefinder:
            pass
        else:
            ax.plot(
                [cluster["thresh"], cluster["thresh"]],
                [0, np.max(clusters_n)],
                c="k",
                label="Threshold",
            )
            ax.legend()

        ax.plot(cuts, clusters_n, **kwargs)
        ax.set_xlabel("Distance Threshold")
        ax.set_ylabel("Number of Clusters")
        plt.tight_layout(**layout_pad)
        return ax, fig

