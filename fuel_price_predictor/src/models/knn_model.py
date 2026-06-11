"""K-Nearest Neighbours regression model for fuel-price prediction.

Wraps a scikit-learn :class:`~sklearn.neighbors.KNeighborsRegressor` tuned with
:class:`~sklearn.model_selection.GridSearchCV` and exposes a small, consistent
API shared with the SVM and Random Forest models.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from ._common import (
    compute_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class _CountryBooster(BaseEstimator, TransformerMixin):
    """Multiply the label-encoded ``country`` column by a fixed weight.

    The ``country`` feature explains ~90 % of the petrol-price variance, so for a
    distance-based learner the ideal behaviour is: pick neighbours from the *same*
    country first, then rank them by the (already-scaled) economic/time features.

    Amplifying this single column makes same-country matching dominate the
    distance and removes the residual cross-country leakage that happens when the
    nearest same-country point is far away in time (its accumulated distance on
    the other features could otherwise lose to an alphabetically adjacent country
    that sits one unit away). Because the column is constant within a country, the
    weight never distorts the within-country interpolation -- it only raises the
    cost of crossing into a different country.

    This is implemented as a tiny scikit-learn transformer so it lives *inside*
    the cross-validated :class:`~sklearn.pipeline.Pipeline` and is serialised with
    the model (no separate state to manage at inference time).
    """

    def __init__(self, country_index: int, weight: float = 10.0) -> None:
        self.country_index = country_index
        self.weight = weight

    def fit(self, X, y=None):  # noqa: D102 - stateless transformer
        return self

    def transform(self, X):  # noqa: D102
        X = np.asarray(X, dtype=float).copy()
        X[:, self.country_index] *= self.weight
        return X


class KNNModel:
    """K-Nearest Neighbours regressor with grid-searched hyper-parameters.

    Parameters
    ----------
    cv : int, default 5
        Number of cross-validation folds used by ``GridSearchCV``.
    n_jobs : int, default -1
        Parallelism passed to ``GridSearchCV`` (``-1`` uses all cores).
    """

    NAME = "KNN"

    #: Hyper-parameter grid explored during :meth:`train`. Keys are prefixed
    #: with ``knn__`` to address the ``KNeighborsRegressor`` step inside the
    #: :class:`~sklearn.pipeline.Pipeline`. The neighbour count is sampled more
    #: finely than the original [3,5,7,10,15,20] grid (k=4 wins on this dataset).
    PARAM_GRID: Dict[str, List] = {
        "knn__n_neighbors": [3, 4, 5, 7, 9, 11, 15, 20],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    #: Default amplification applied to the ``country`` column (see
    #: :class:`_CountryBooster`). Any value >= ~5 saturates on this dataset.
    COUNTRY_WEIGHT = 10.0

    def __init__(self, cv: int = 5, n_jobs: int = -1) -> None:
        """Create an untrained KNN model."""
        self.cv = cv
        self.n_jobs = n_jobs
        self.model: KNeighborsRegressor | None = None
        self.best_params_: Dict | None = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        country_index: int | None = None,
        country_weight: float | None = None,
    ) -> "KNNModel":
        """Run ``GridSearchCV`` over a KNN pipeline and keep the best estimator.

        Counter-intuitively for a distance-based model, the raw (mostly unscaled)
        design matrix is *better* here than a uniformly ``StandardScaler``-d one:
        the label-encoded ``country`` column dominating the distance is exactly
        what makes KNN pick same-country neighbours, which is ideal because
        ``country`` explains ~90 % of the price. Standardising every feature breaks
        that and roughly triples the RMSE.

        The only genuinely helpful tweak is to *strengthen* that signal. When
        *country_index* is given the column is amplified by a :class:`_CountryBooster`
        inside the cross-validated pipeline, removing the residual cross-country
        leakage; the booster is constant within a country so it never disturbs the
        within-country interpolation. Pass ``country_index=None`` to train on the
        matrix as-is (the model still works, just without the boost).

        Logs the chosen hyper-parameters and the (RMSE-converted) CV score.
        """
        weight = self.COUNTRY_WEIGHT if country_weight is None else country_weight
        steps = []
        if country_index is not None:
            steps.append(("boost", _CountryBooster(country_index, weight)))
        steps.append(("knn", KNeighborsRegressor()))
        pipe = Pipeline(steps)

        n_combos = (
            len(self.PARAM_GRID["knn__n_neighbors"])
            * len(self.PARAM_GRID["knn__weights"])
            * len(self.PARAM_GRID["knn__metric"])
        )
        logger.info("[KNN] Starting GridSearchCV over %d combinations "
                    "(country boost=%s)...", n_combos,
                    f"x{weight:g}" if country_index is not None else "off")
        search = GridSearchCV(
            estimator=pipe,
            param_grid=self.PARAM_GRID,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            n_jobs=self.n_jobs,
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        # Strip the ``knn__`` pipeline prefix so the comparison JSON, the train.py
        # summary table and the Gradio comparison tab keep reading cleanly.
        self.best_params_ = {
            key.split("__", 1)[-1]: val
            for key, val in search.best_params_.items()
        }
        self.best_score_ = float(search.best_score_)
        cv_rmse = float(np.sqrt(-self.best_score_))

        logger.info("[KNN] Best params: %s", self.best_params_)
        logger.info("[KNN] Best CV score: MSE=%.5f (RMSE=%.5f)",
                    -self.best_score_, cv_rmse)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return petrol-price predictions for the design matrix *X*."""
        self._check_trained()
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Return ``{MAE, MSE, RMSE, R2, MAPE}`` on the held-out test set."""
        self._check_trained()
        metrics = compute_metrics(y_test, self.predict(X_test))
        logger.info("[KNN] Test metrics: %s",
                    {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def get_best_params(self) -> Dict:
        """Return the best hyper-parameters found by ``GridSearchCV``."""
        self._check_trained()
        return dict(self.best_params_)

    # ------------------------------------------------------------------ #
    # Diagnostic plots
    # ------------------------------------------------------------------ #
    def plot_predictions_vs_actual(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: str,
    ) -> str:
        """Save a predicted-vs-actual scatter plot to *save_path*."""
        self._check_trained()
        return plot_predictions_vs_actual(
            y_test, self.predict(X_test), self.NAME, save_path)

    def plot_residuals(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: str,
    ) -> str:
        """Save a residual plot to *save_path*."""
        self._check_trained()
        return plot_residuals(y_test, self.predict(X_test), self.NAME, save_path)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialise the trained model (and metadata) via joblib."""
        self._check_trained()
        joblib.dump(
            {
                "model": self.model,
                "best_params_": self.best_params_,
                "best_score_": self.best_score_,
            },
            path,
            compress=3,
        )
        logger.info("[KNN] Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "KNNModel":
        """Load a previously saved KNN model from *path*."""
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params_")
        obj.best_score_ = payload.get("best_score_")
        logger.info("[KNN] Loaded model from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("KNNModel is not trained yet. Call train() first.")
