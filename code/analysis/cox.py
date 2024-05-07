from typing import List
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import pandas as pd
import statsmodels.api as sm
from group_lasso import LogisticGroupLasso

# for CoxModelUtil
from lifelines import CoxPHFitter
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin

# for LogitModelUtil
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, check_X_y
from statsmodels.discrete.discrete_model import Logit

from .abstractions import ModelUtil, ClinicalModel, Status


class CoxModelUtil(ModelUtil):
    """Standard cox model based on Lifelines CoxPHFitter."""

    def __init__(self, **kwargs):
        self.model = CoxPHFitter()
        self.step_size = 0.1
        self.__dict__.update(kwargs)
        self.STATUS = Status.OK

    def _fit_model(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    self.train_df, self.time_col, self.event_col, step_size=self.step_size
                )
                self.STATUS = Status.OK
        except Exception:
            warnings.warn("regression failed")
            self.STATUS = Status.FAIL

        if self.STATUS == Status.OK:
            self.fit_params = self.model.summary[["exp(coef)", "p"]]
            self.fit_params.rename(columns={"exp(coef)": "hr"})
            self.fit_params["nlp"] = self.fit_params["p"].apply(np.log10)
            self.fit_scores["ci"] = self.model.concordance_index_
        elif self.STATUS == Status.FAIL:
            self.fit_params = pd.DataFrame(
                {"hr": [None] * self.nvars, "p": [None] * self.nvars, "nlp": [None] * self.nvars},
                index=self.var_list,
            )
            self.fit_scores["ci"] = None

    def fit(self, train_df: pd.DataFrame, time_col: str = None, event_col: str = None) -> None:

        assert time_col is not None, "time_col not defined"
        assert time_col in train_df, "time_col not found in train_df"
        self.time_col = time_col

        assert event_col is not None, "event_col not defined"
        assert event_col in train_df, "event_col not found in train_df"
        self.event_col = event_col

        self.train_df = train_df
        self.var_list = [v for v in train_df.columns if v not in [time_col, event_col]]
        self.nvars = len(self.var_list)
        self.fit_scores = {}
        self._fit_model()

    def predict(
        self, test_df: pd.DataFrame, test_ci: bool = False, test_auc_at_time: float = None
    ) -> None:

        assert self.STATUS == Status.OK, "model training failed, cannot test"

        self.linear_pred = self.model.predict_log_partial_hazard(test_df)
        self.test_scores = {}

        if test_ci:
            self.test_scores["ci"] = self.model.score(test_df, scoring_method="concordance_index")

        if test_auc_at_time is not None:
            # TODO: add test AUC for a particular time, e.g. at 6 MO
            self.test_scores["auc_at_time"] = 0.5



class CoxModel(ClinicalModel):
    def __init__(
        self,
        name,
        time_col: str,
        event_col: str,
        arm_col: str = None,
        cov_cols: List[str] = None,
        test_ci: bool = False,
        test_auc_at_time: float = None,
    ):
        """Provide self with some cox model characteristics.

        The event_col takes a values of 1 if the event was observed, or a 0 if
        censored, in alignment with lifelines.
        See: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html#lifelines.fitters.coxph_fitter.CoxPHFitter.predict_cumulative_hazard
        """
        super().__init__(name)
        self.model_info = dict()
        self.model_info["form"] = "cox"
        self.time_col = time_col
        self.event_col = event_col
        self.arm_col = arm_col
        self.test_ci = test_ci
        self.test_auc_at_time = test_auc_at_time
        if cov_cols is None:
            cov_cols = []
        self.cov_cols = cov_cols

    def _run_model(self, model_df: pd.DataFrame) -> None:

        if self.train_idx is not None:
            train_df = model_df[model_df.index.isin(self.train_idx)]
        else:
            train_df = model_df

        if self.test_idx is not None:
            test_df = model_df[model_df.index.isin(self.test_idx)]
        else:
            test_df = None

        self.cm = CoxModelUtil()
        self.cm.fit(train_df, time_col=self.time_col, event_col=self.event_col)
        if test_df is not None:
            self.cm.predict(test_df, test_ci=self.test_ci, test_auc_at_time=self.test_auc_at_time)

    def run_prognostic(
        self, hif: str, arm: str = None, train_idx: List = None, test_idx: List = None
    ) -> None:
        """Fit prognostic model to single arm or all data."""
        self.model_info["mode"] = "prognostic"
        self.train_idx = train_idx
        self.test_idx = test_idx
        if arm is not None:
            assert self.arm_col is not None, "arm_col, arm column name, undefined"
            model_df = self.df[self.df[self.arm_col] == arm].copy()
        else:
            model_df = self.df.copy()
        model_cols = [hif, self.time_col, self.event_col] + self.cov_cols
        self._run_model(model_df[model_cols])

    def run_interaction(
        self,
        hif: str,
        target_arm: str,
        reference_arm: str,
        train_idx: List = None,
        test_idx: List = None,
    ) -> None:
        """Fit treatment-feature interaction model."""

        # TODO: add checks for inputs, etc
        # TODO: add useful / standard reporting of model outputs
        self.model_info["mode"] = "prognostic"
        self.train_idx = train_idx
        self.test_idx = test_idx

        assert self.arm_col is not None, "arm_col, arm column name, undefined"

        model_df = self.df[self.df[self.arm_col].isin([target_arm, reference_arm])].copy()

        treat_name = f"{target_arm}_{reference_arm}"
        inter_name = f"{target_arm}x{hif}"

        model_df.loc[:, treat_name] = 1 * (model_df[self.arm_col].isin([target_arm]))
        model_df.loc[:, inter_name] = model_df[treat_name] * model_df[hif]

        model_cols = [hif, treat_name, inter_name, self.time_col, self.event_col] + self.cov_cols

        self._run_model(model_df[model_cols])

    def run_treatment_effect_enrichment(
        self,
        hif: str,
        target_arm: str,
        reference_arm: str,
        train_idx: List = None,
        test_idx: List = None,
    ) -> None:
        """Fit treatment effect enrichment model."""
        self.model_info["mode"] = "treatment_effect_enrichment"
        self.train_idx = train_idx
        self.test_idx = test_idx

        model_df = self.df[self.df[self.arm_col].isin([target_arm, reference_arm])].copy()

        model_df = model_df[model_df[hif] == 1]

        treat_name = f"{target_arm}_{reference_arm}"

        model_df.loc[:, treat_name] = 1 * (model_df[self.arm_col].isin([target_arm]))
        model_cols = [treat_name, self.time_col, self.event_col] + self.cov_cols

        self._run_model(model_df[model_cols])

    @classmethod
    def from_datasets(
        cls,
        feature_df: pd.DataFrame,  # clinical_df: pd.DataFrame,
        name,
        time_col: str,
        event_col: str,
        arm_col: str = None,
        cov_cols: List[str] = None,
        test_ci: bool = False,
        test_auc_at_time: float = None,
    ) -> "CoxModel":
        """Create CoxModel from feature and clinical data sets."""
        obj = cls(
            name,
            time_col,
            event_col,
            cov_cols=cov_cols,
            arm_col=arm_col,
            test_ci=test_ci,
            test_auc_at_time=test_auc_at_time,
        )
        obj.attach_features_df(feature_df)
        return obj
