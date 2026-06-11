"""Support Vector Regression model for fuel-price prediction.

Tunes a Support Vector regressor over the full ``rbf`` / ``linear`` / ``poly``
grid and exposes the shared model API. Two performance-critical decisions make
this tractable on a large dataset:

1. **`SVR(kernel='linear')` is replaced by** :class:`~sklearn.svm.LinearSVR`.
   The libsvm linear kernel is ~O(n²–n³) and is pathologically slow here
   (a single fit on 1–3k rows can take minutes); liblinear's ``LinearSVR`` is
   ~O(n) and fits 10k rows in a few seconds while optimising the same
   epsilon-insensitive objective.
2. **Tune small, refit big.** Hyper-parameters are grid-searched (cv=5) on a
   small *tuning* subset (default 2,000 rows); the single best configuration is
   then refit on the full subsample (default 10,000 rows). This keeps the search
   cheap without sacrificing the final model's data size.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR

from ._common import (
    compute_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class SVMModel:
    """Support Vector regressor with grid-searched hyper-parameters.

    Parameters
    ----------
    cv : int, default 5
        Cross-validation folds for ``GridSearchCV``.
    n_jobs : int, default -1
        Parallelism for ``GridSearchCV``.
    """

    NAME = "SVM"

    #: Hyper-parameter grid explored during :meth:`train` (spec reference).
    #: The ``linear`` kernel is routed through :class:`LinearSVR` for speed
    #: (where ``gamma`` does not apply).
    PARAM_GRID: Dict[str, List] = {
        "kernel": ["rbf", "linear", "poly"],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "gamma": ["scale", "auto"],
    }

    #: Iteration cap for liblinear (keeps the linear search bounded).
    LINEAR_MAX_ITER = 10_000

    #: Iteration cap for libsvm rbf/poly. Without it, poly/rbf with large ``C``
    #: can fail to converge and run almost unbounded; this guarantees each fit
    #: terminates (a slightly under-converged fit is fine for model selection).
    SVR_MAX_ITER = 5_000

    def __init__(self, cv: int = 5, n_jobs: int = -1) -> None:
        """Create an untrained SVM model."""
        self.cv = cv
        self.n_jobs = n_jobs
        self.model = None
        self.best_params_: Dict | None = None
        self.best_score_: float | None = None
        #: Human-readable note describing the subsample / tuning strategy.
        self.subsample_info: str = ""

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        subsample: bool = True, subsample_size: int = 10_000,
        tune_size: int = 1_500,
    ) -> "SVMModel":
        """Grid-search hyper-parameters then refit the best model.

        Parameters
        ----------
        X_train, y_train : numpy.ndarray
            Full training design matrix and target.
        subsample : bool, default True
            When True and the training set exceeds *subsample_size*, the final
            model is fit on a random *subsample_size* subset (``random_state=42``).
        subsample_size : int, default 10000
            Maximum number of rows used to fit the final model.
        tune_size : int, default 2000
            Rows used for the (cv-folded) hyper-parameter search. Much smaller
            than *subsample_size* so the 72-combination grid stays fast.
        """
        y_train = np.asarray(y_train, dtype=float)
        rng = np.random.RandomState(RANDOM_STATE)
        n = len(X_train)

        # --- Final-fit subsample ------------------------------------- #
        if subsample and n > subsample_size:
            fit_idx = rng.choice(n, size=subsample_size, replace=False)
            X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
            sub_note = (f"final fit on {subsample_size:,} of {n:,} rows "
                        f"(random_state={RANDOM_STATE})")
        else:
            X_fit, y_fit = X_train, y_train
            sub_note = f"final fit on all {n:,} rows"

        # --- Tuning subset (subset of the final-fit data) ------------ #
        m = len(X_fit)
        if m > tune_size:
            tune_idx = rng.choice(m, size=tune_size, replace=False)
            X_tune, y_tune = X_fit[tune_idx], y_fit[tune_idx]
        else:
            X_tune, y_tune = X_fit, y_fit

        cs = self.PARAM_GRID["C"]
        eps = self.PARAM_GRID["epsilon"]
        gammas = self.PARAM_GRID["gamma"]

        # The max_iter caps intentionally produce ConvergenceWarnings; silence
        # them locally so the training log stays readable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            # --- Search A: SVR over rbf + poly (kernels needing gamma) --- #
            logger.info("[SVM] Tuning rbf/poly via SVR on %d rows (cv=%d)...",
                        len(X_tune), self.cv)
            svr_grid = {"kernel": ["rbf", "poly"], "C": cs,
                        "epsilon": eps, "gamma": gammas}
            svr_search = GridSearchCV(
                SVR(cache_size=1000, max_iter=self.SVR_MAX_ITER), svr_grid,
                cv=self.cv, scoring="neg_mean_squared_error",
                n_jobs=self.n_jobs, refit=False)
            svr_search.fit(X_tune, y_tune)
            logger.info("[SVM] Best rbf/poly: %s (MSE=%.5f)",
                        svr_search.best_params_, -svr_search.best_score_)

            # --- Search B: LinearSVR (the fast 'linear' kernel) ---------- #
            logger.info("[SVM] Tuning linear via LinearSVR on %d rows (cv=%d)...",
                        len(X_tune), self.cv)
            lin_grid = {"C": cs, "epsilon": eps}
            lin_search = GridSearchCV(
                LinearSVR(max_iter=self.LINEAR_MAX_ITER,
                          random_state=RANDOM_STATE),
                lin_grid, cv=self.cv, scoring="neg_mean_squared_error",
                n_jobs=self.n_jobs, refit=False)
            lin_search.fit(X_tune, y_tune)
            logger.info("[SVM] Best linear: %s (MSE=%.5f)",
                        lin_search.best_params_, -lin_search.best_score_)

            # --- Pick the better search (least negative MSE) ------------- #
            if svr_search.best_score_ >= lin_search.best_score_:
                params = dict(svr_search.best_params_)
                self.best_score_ = float(svr_search.best_score_)
                self.model = SVR(cache_size=1000,
                                 max_iter=self.SVR_MAX_ITER, **params)
                self.best_params_ = params
            else:
                params = dict(lin_search.best_params_)
                self.best_score_ = float(lin_search.best_score_)
                self.model = LinearSVR(max_iter=self.LINEAR_MAX_ITER,
                                       random_state=RANDOM_STATE, **params)
                # Record a 'kernel' key so downstream display is consistent.
                self.best_params_ = {"kernel": "linear", **params}

            # --- Refit the winner on the full subsample ------------------ #
            logger.info("[SVM] Refitting best model (%s) on %d rows...",
                        self.best_params_, len(X_fit))
            self.model.fit(X_fit, y_fit)

        self.subsample_info = (
            f"Tuned 72-combo grid on {len(X_tune):,}-row subset (cv={self.cv}); "
            f"linear kernel via LinearSVR; {sub_note}."
        )
        cv_rmse = float(np.sqrt(-self.best_score_))
        logger.info("[SVM] %s", self.subsample_info)
        logger.info("[SVM] Best params: %s", self.best_params_)
        logger.info("[SVM] Best CV score: MSE=%.5f (RMSE=%.5f)",
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
        logger.info("[SVM] Test metrics: %s",
                    {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def get_best_params(self) -> Dict:
        """Return the best hyper-parameters found during tuning."""
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
                "subsample_info": self.subsample_info,
            },
            path,
            compress=3,
        )
        logger.info("[SVM] Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "SVMModel":
        """Load a previously saved SVM model from *path*."""
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params_")
        obj.best_score_ = payload.get("best_score_")
        obj.subsample_info = payload.get("subsample_info", "")
        logger.info("[SVM] Loaded model from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("SVMModel is not trained yet. Call train() first.")
