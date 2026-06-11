"""Random Forest regression model for fuel-price prediction.

Wraps a scikit-learn :class:`~sklearn.ensemble.RandomForestRegressor` tuned with
:class:`~sklearn.model_selection.RandomizedSearchCV`. In addition to the shared
model API it exposes :meth:`plot_feature_importance` (exclusive to this model).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from ._common import (
    CORAL,
    DEEP_GREEN,
    PLOT_DPI,
    PRIMARY,
    compute_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class RandomForestModel:
    """Random Forest regressor tuned via ``RandomizedSearchCV``.

    Parameters
    ----------
    n_iter : int, default 30
        Number of parameter settings sampled by ``RandomizedSearchCV``.
    cv : int, default 5
        Cross-validation folds.
    n_jobs : int, default -1
        Parallelism for the search and the forest.
    """

    NAME = "Random Forest"

    #: Parameter distribution sampled during :meth:`train`.
    PARAM_DIST: Dict[str, List] = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    def __init__(self, n_iter: int = 30, cv: int = 5, n_jobs: int = -1) -> None:
        """Create an untrained Random Forest model."""
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.model: RandomForestRegressor | None = None
        self.best_params_: Dict | None = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        """Run ``RandomizedSearchCV`` and keep the best estimator."""
        logger.info("[RF] Starting RandomizedSearchCV (n_iter=%d)...", self.n_iter)
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE),
            param_distributions=self.PARAM_DIST,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            n_jobs=self.n_jobs,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)
        cv_rmse = float(np.sqrt(-self.best_score_))

        logger.info("[RF] Best params: %s", self.best_params_)
        logger.info("[RF] Best CV score: MSE=%.5f (RMSE=%.5f)",
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
        logger.info("[RF] Test metrics: %s",
                    {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def get_best_params(self) -> Dict:
        """Return the best hyper-parameters found by ``RandomizedSearchCV``."""
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

    def plot_feature_importance(
        self, feature_names: List[str], save_path: str, top_n: int = 15,
    ) -> str:
        """Save a horizontal bar chart of the top-*n* feature importances.

        This diagnostic is exclusive to the Random Forest model, which natively
        exposes impurity-based ``feature_importances_``.

        Parameters
        ----------
        feature_names : list of str
            Names matching the columns of the training design matrix.
        save_path : str
            Destination PNG path.
        top_n : int, default 15
            Number of highest-importance features to display.
        """
        self._check_trained()
        from ._common import _apply_style

        importances = np.asarray(self.model.feature_importances_, dtype=float)
        names = np.asarray(feature_names)
        if len(names) != len(importances):
            raise ValueError(
                f"feature_names length ({len(names)}) does not match the model's "
                f"feature count ({len(importances)})."
            )

        order = np.argsort(importances)[::-1][:top_n]
        top_names = names[order][::-1]      # reversed for ascending barh display
        top_values = importances[order][::-1]

        plt = _apply_style()
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(range(len(top_values)), top_values, color=DEEP_GREEN,
                       edgecolor=PRIMARY, linewidth=0.6)
        # Highlight the single most important feature in coral.
        bars[-1].set_color(CORAL)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, color=PRIMARY)
        ax.set_xlabel("Feature importance (impurity-based)", color=PRIMARY)
        ax.set_title(f"Random Forest — Top {len(top_names)} Feature Importances",
                     color=PRIMARY, fontsize=13, fontweight="bold")
        for i, v in enumerate(top_values):
            ax.text(v + max(top_values) * 0.01, i, f"{v:.3f}",
                    va="center", color=PRIMARY, fontsize=8)
        fig.tight_layout()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("[RF] Saved feature-importance chart to %s", save_path)
        return save_path

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialise the trained model (and metadata) via joblib."""
        self._check_trained()
        # compress=3: the tuned forest (300 fully-grown trees) is ~375 MB raw;
        # compression shrinks it ~4x for a deployable artifact, unchanged model.
        joblib.dump(
            {
                "model": self.model,
                "best_params_": self.best_params_,
                "best_score_": self.best_score_,
            },
            path,
            compress=3,
        )
        logger.info("[RF] Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        """Load a previously saved Random Forest model from *path*."""
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params_")
        obj.best_score_ = payload.get("best_score_")
        logger.info("[RF] Loaded model from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "RandomForestModel is not trained yet. Call train() first.")
