"""Shared utilities for the regression models.

Centralises metric computation and the diagnostic plots so the three model
classes (:class:`KNNModel`, :class:`SVMModel`, :class:`RandomForestModel`)
expose an identical, design-system-consistent API without copy-paste drift.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Design-system palette (Cohere-inspired) reused across every chart.
# ---------------------------------------------------------------------------
PRIMARY = "#17171c"      # near-black
DEEP_GREEN = "#003c33"   # dark band
CORAL = "#ff7759"        # accent
ACTION_BLUE = "#1863dc"  # info / links
MUTED = "#93939f"        # secondary text
HAIRLINE = "#d9d9dd"     # borders
PLOT_DPI = 150


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return the standard regression metric dict.

    Keys: ``MAE``, ``MSE``, ``RMSE``, ``R2`` and ``MAPE`` (the latter as a
    percentage). All values are plain Python floats for easy JSON serialising.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


def _apply_style():
    """Apply the project plot style and return the matplotlib.pyplot module.

    Imported lazily so that importing a model class never hard-requires
    matplotlib (handy on minimal inference environments).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless / server-safe backend
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:  # older matplotlib naming fallback
        plt.style.use("seaborn-whitegrid")
    return plt


def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str,
) -> str:
    """Scatter predicted vs. actual values with an ideal ``y = x`` reference."""
    plt = _apply_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=14, alpha=0.45, color=DEEP_GREEN,
               edgecolors="none", label="Predictions")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=CORAL, lw=2, ls="--", label="Ideal (y = x)")

    metrics = compute_metrics(y_true, y_pred)
    ax.set_title(f"{model_name} — Predicted vs Actual (R² = {metrics['R2']:.3f})",
                 color=PRIMARY, fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual petrol price (USD/L)", color=PRIMARY)
    ax.set_ylabel("Predicted petrol price (USD/L)", color=PRIMARY)
    ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str,
) -> str:
    """Plot residuals (actual − predicted) against predicted values."""
    plt = _apply_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_pred, residuals, s=14, alpha=0.45, color=ACTION_BLUE,
               edgecolors="none")
    ax.axhline(0.0, color=CORAL, lw=2, ls="--")
    ax.set_title(f"{model_name} — Residual Plot", color=PRIMARY,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted petrol price (USD/L)", color=PRIMARY)
    ax.set_ylabel("Residual (Actual − Predicted)", color=PRIMARY)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path
