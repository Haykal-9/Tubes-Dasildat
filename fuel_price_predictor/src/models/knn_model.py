"""Scaled K-Nearest Neighbours model for fuel-price prediction."""

from __future__ import annotations

import logging
from typing import Dict, List

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._common import compute_metrics, plot_predictions_vs_actual, plot_residuals

logger = logging.getLogger(__name__)


class KNNModel:
    """KNN with StandardScaler inside the persisted sklearn Pipeline."""

    NAME = "KNN"
    PARAM_GRID: Dict[str, List] = {
        "knn__n_neighbors": [3, 5, 7, 10, 15, 20],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    def __init__(self, cv: int = 5, n_jobs: int = -1) -> None:
        self.cv = cv
        self.n_jobs = n_jobs
        self.model: Pipeline | None = None
        self.best_params_: Dict | None = None
        self.best_score_: float | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNModel":
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsRegressor()),
        ])
        search = GridSearchCV(
            pipeline, self.PARAM_GRID, cv=self.cv,
            scoring="neg_mean_squared_error", n_jobs=self.n_jobs,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params_ = {
            key.split("__", 1)[-1]: value
            for key, value in search.best_params_.items()
        }
        self.best_score_ = float(search.best_score_)
        logger.info("[KNN] Best params: %s", self.best_params_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        metrics = compute_metrics(y_test, self.predict(X_test))
        logger.info("[KNN] Test metrics: %s",
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
        }, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "KNNModel":
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params_")
        obj.best_score_ = payload.get("best_score_")
        return obj

    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("KNNModel is not trained yet. Call train() first.")
