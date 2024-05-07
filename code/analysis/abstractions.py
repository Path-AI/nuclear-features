from typing import List
from enum import Enum, auto

import pandas as pd
from abc import ABC, abstractmethod

from .utils import robust_z

class Status(Enum):
    OK = auto()
    FAIL = auto()

class ModelUtil(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> None:
        pass

class Featureset:
    """Class for generic manipulations of a featureset. Used as the parent of cluster and classification."""

    def __init__(self, name: str) -> None:
        """Initiatilization of clustering, only takes a name. Initializes a properties dict for tracking what the
        clustering corresponds to."""
        self.general_properties = {"name": name}

    def attach_features_df(self, df: pd.DataFrame) -> None:
        """Attaches a pandas dataframe for use in clustering. Each column is assumed to be a feature adn each row is
        assumed to be a sample.
        """
        self.df = df.copy()
        self.general_properties["indexed_by"] = df.index.name


class ClinicalModel(Featureset):
    """Generic class for constructing clinical models from featuresets. As such, inherits from Featureset."""

    def __init__(self, name):
        super().__init__(name)
        # TODO: what else should go here?

    def attach_dependent_df(
        self, y: pd.DataFrame, y_names: List[str], index_names: List[str] = ["slideId", "caseName"]
    ) -> None:
        """Attaches a dataset that we seek to relate the features to, can include any list of possible outcome cols."""
        feature_df = self.df.copy()

        assert (
            y.index.name == feature_df.index.name
        ), "Indexing must be same between dependent df and featureset."

        # combine dfs, remove duplicates, remove indexes
        combined_df = pd.DataFrame(y[y_names]).join(feature_df, how="inner")
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        X_names = feature_df.columns.tolist()
        for inn in index_names:
            try:
                X_names.remove(inn)
            except ValueError:
                pass
        self.X_names = X_names
        self.y_names = y_names
        self.df = combined_df

    def qc(self, **kwargs) -> None:
        """Augments standard qc by applying qc to combined dataframe, updates potential X and y names"""
        assert hasattr(self, "y_names"), "Both X and y must be added."
        super().qc(**kwargs)

        surviving_cols = self.df.columns.tolist()
        self.y_names = [oc for oc in self.y_names if oc in surviving_cols]
        self.X_names = [sc for sc in surviving_cols if sc not in self.y_names]

    def correlate_X(self, **kwargs) -> None:
        """Correlates features ignoring the dependent variables. Replaces Featureset function."""
        assert hasattr(self, "y_names"), "Dependent df must be attached."
        self.Xcorr = self.df[self.X_names].corr(**kwargs)

    def get_X(self) -> pd.DataFrame:
        """Returns dataframe of features excluding dependent variables."""
        assert hasattr(self, "y_names"), "Dependent df must be attached."
        return self.df[self.X_names]

    def get_norm_X(self, norm="z"):
        """Returns dataframe of z-scored features excluding dependent variables."""
        assert hasattr(self, "y_names"), "Dependent df must be attached."
        assert norm in ["z", "robust_z"], f"{norm} not implemented."
        if norm == "z":
            return self.df[self.X_names].apply(stats.zscore, ddof=1)
        elif norm == "robust_z":
            return robust_z([self.X_names])