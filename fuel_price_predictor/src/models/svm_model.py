"""Scaled Support Vector Regression model for fuel-price prediction."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

from ._common import (
    compute_metrics,
    make_time_series_cv,
    plot_predictions_vs_actual,
    plot_residuals,
)

logger = logging.getLogger(__name__)
RANDOM_STATE = 42


class SVMModel:
    """SVR with StandardScaler inside every tuned and persisted Pipeline."""

    NAME = "SVM"
    PARAM_GRID: Dict[str, List] = {
        "kernel": ["rbf", "linear", "poly"],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "gamma": ["scale", "auto"],
    }
    LINEAR_MAX_ITER = 10_000
    SVR_MAX_ITER = 5_000

    def __init__(self, cv: int = 5, n_jobs: int = -1) -> None:
        self.cv = cv
        self.n_jobs = n_jobs
        self.model: Pipeline | None = None
        self.best_params_: Dict | None = None
        self.best_score_: float | None = None
        self.subsample_info = ""
        self.cv_strategy = "TimeSeriesSplit"

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        subsample: bool = True, subsample_size: int = 10_000,
        tune_size: int = 1_500,
    ) -> "SVMModel":
        n = len(X_train)
        if subsample and n > subsample_size:
            X_fit, y_fit = X_train[-subsample_size:], y_train[-subsample_size:]
            fit_note = (
                f"final fit on most recent {subsample_size:,} of {n:,} rows"
            )
        else:
            X_fit, y_fit = X_train, y_train
            fit_note = f"final fit on all {n:,} rows"

        if len(X_fit) > tune_size:
            X_tune, y_tune = X_fit[-tune_size:], y_fit[-tune_size:]
        else:
            X_tune, y_tune = X_fit, y_fit
        temporal_cv = make_time_series_cv(len(X_tune), self.cv)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            svr_pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("svr", SVR(cache_size=1000, max_iter=self.SVR_MAX_ITER)),
            ])
            svr_grid = {
                "svr__kernel": ["rbf", "poly"],
                "svr__C": self.PARAM_GRID["C"],
                "svr__epsilon": self.PARAM_GRID["epsilon"],
                "svr__gamma": self.PARAM_GRID["gamma"],
            }
            svr_search = GridSearchCV(
                svr_pipeline, svr_grid, cv=temporal_cv,
                scoring="neg_mean_squared_error", n_jobs=self.n_jobs, refit=False,
            )
            svr_search.fit(X_tune, y_tune)

            linear_pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("svr", LinearSVR(
                    max_iter=self.LINEAR_MAX_ITER, random_state=RANDOM_STATE)),
            ])
            linear_grid = {
                "svr__C": self.PARAM_GRID["C"],
                "svr__epsilon": self.PARAM_GRID["epsilon"],
            }
            linear_search = GridSearchCV(
                linear_pipeline, linear_grid, cv=temporal_cv,
                scoring="neg_mean_squared_error", n_jobs=self.n_jobs, refit=False,
            )
            linear_search.fit(X_tune, y_tune)

            if svr_search.best_score_ >= linear_search.best_score_:
                raw_params = dict(svr_search.best_params_)
                self.model = Pipeline([
                    ("scale", StandardScaler()),
                    ("svr", SVR(
                        cache_size=1000, max_iter=self.SVR_MAX_ITER,
                        **{key.split("__", 1)[1]: value
                           for key, value in raw_params.items()})),
                ])
                params = {
                    key.split("__", 1)[1]: value
                    for key, value in raw_params.items()
                }
                self.best_score_ = float(svr_search.best_score_)
            else:
                raw_params = dict(linear_search.best_params_)
                params = {
                    key.split("__", 1)[1]: value
                    for key, value in raw_params.items()
                }
                self.model = Pipeline([
                    ("scale", StandardScaler()),
                    ("svr", LinearSVR(
                        max_iter=self.LINEAR_MAX_ITER,
                        random_state=RANDOM_STATE, **params)),
                ])
                params = {"kernel": "linear", **params}
                self.best_score_ = float(linear_search.best_score_)

            self.model.fit(X_fit, y_fit)

        self.best_params_ = params
        self.subsample_info = (
            f"Scaled Pipeline; tuned on most recent {len(X_tune):,} rows "
            f"(cv={self.cv_strategy}, splits={self.cv}); "
            f"{fit_note}."
        )
        logger.info("[SVM] Best params: %s", self.best_params_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        # Linear/poly extrapolation can dip slightly below zero on extreme tax
        # inputs; fuel prices are non-negative by definition.
        return np.maximum(self.model.predict(X), 0.0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        metrics = compute_metrics(y_test, self.predict(X_test))
        logger.info("[SVM] Test metrics: %s",
                    {key: round(value, 4) for key, value in metrics.items()})
        return metrics

    def get_best_params(self) -> Dict:
        self._check_trained()
        return dict(self.best_params_)

    def plot_predictions_vs_actual(self, X_test, y_test, save_path: str) -> str:
        return plot_predictions_vs_actual(
            y_test, self.predict(X_test), self.NAME, save_path)

    def plot_residuals(self, X_test, y_test, save_path: str) -> str:
        return plot_residuals(y_test, self.predict(X_test), self.NAME, save_path)

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump({
            "model": self.model,
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "subsample_info": self.subsample_info,
            "cv_strategy": self.cv_strategy,
        }, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "SVMModel":
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params_")
        obj.best_score_ = payload.get("best_score_")
        obj.subsample_info = payload.get("subsample_info", "")
        obj.cv_strategy = payload.get("cv_strategy", "TimeSeriesSplit")
        return obj

    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("SVMModel is not trained yet. Call train() first.")
