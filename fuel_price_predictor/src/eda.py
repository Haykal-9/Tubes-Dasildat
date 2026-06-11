"""Exploratory-data-analysis helpers shared by the trainer and the web app.

Produces the dataset-overview charts displayed in the Gradio "Dataset Overview"
tab and a descriptive-statistics table. Kept separate so plots can be
re-generated at app start-up (they are git-ignored).
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from .models._common import (
    ACTION_BLUE,
    CORAL,
    DEEP_GREEN,
    PLOT_DPI,
    PRIMARY,
    _apply_style,
)

logger = logging.getLogger(__name__)

TARGET = "petrol_usd_liter"

# Region -> consistent palette colour used in the region chart.
_REGION_PALETTE = [
    DEEP_GREEN, CORAL, ACTION_BLUE, PRIMARY, "#6a994e", "#9b5de5", "#f4a261",
]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns.

    The frame is transposed so each row is a column and the statistics
    (``count, mean, std, min, max`` and quartiles) are the columns -- a layout
    that renders cleanly in ``gr.Dataframe``.
    """
    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T.reset_index().rename(columns={"index": "feature"})
    for col in stats.columns:
        if col != "feature":
            stats[col] = stats[col].round(3)
    return stats


def plot_target_distribution(df: pd.DataFrame, save_path: str) -> str:
    """Histogram (+KDE-like density) of ``petrol_usd_liter``."""
    plt = _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    values = df[TARGET].astype(float)
    ax.hist(values, bins=40, color=DEEP_GREEN, alpha=0.85, edgecolor="white")
    ax.axvline(values.mean(), color=CORAL, lw=2, ls="--",
               label=f"Mean = {values.mean():.2f}")
    ax.axvline(values.median(), color=ACTION_BLUE, lw=2, ls=":",
               label=f"Median = {values.median():.2f}")
    ax.set_title("Distribution of Petrol Price (USD/L)", color=PRIMARY,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Petrol price (USD/L)", color=PRIMARY)
    ax.set_ylabel("Frequency", color=PRIMARY)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_price_timeseries(df: pd.DataFrame, save_path: str) -> str:
    """Line chart of the global monthly-average petrol price over time."""
    plt = _apply_style()
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])
    monthly = (
        tmp.set_index("date")[TARGET]
        .resample("ME").mean()
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, color=DEEP_GREEN, lw=2)
    ax.fill_between(monthly.index, monthly.values, color=DEEP_GREEN, alpha=0.12)
    ax.set_title("Global Monthly-Average Petrol Price (2020–2026)",
                 color=PRIMARY, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", color=PRIMARY)
    ax.set_ylabel("Avg petrol price (USD/L)", color=PRIMARY)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_region_distribution(df: pd.DataFrame, save_path: str) -> str:
    """Box plot of petrol price grouped by region (sorted by median)."""
    plt = _apply_style()
    regions = (
        df.groupby("region")[TARGET].median().sort_values(ascending=True).index
    )
    data = [df.loc[df["region"] == r, TARGET].astype(float).values for r in regions]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=list(regions),
                    medianprops=dict(color=PRIMARY, linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=CORAL, alpha=0.4,
                                    markeredgecolor="none"))
    for patch, color in zip(bp["boxes"], _REGION_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("Petrol Price Distribution by Region", color=PRIMARY,
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Petrol price (USD/L)", color=PRIMARY)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_overview_plots(df: pd.DataFrame, plots_dir: str) -> Dict[str, str]:
    """Generate all three dataset-overview charts and return their paths."""
    logger.info("Generating dataset-overview plots in %s", plots_dir)
    return {
        "distribution": plot_target_distribution(
            df, os.path.join(plots_dir, "petrol_distribution.png")),
        "timeseries": plot_price_timeseries(
            df, os.path.join(plots_dir, "price_timeseries.png")),
        "region": plot_region_distribution(
            df, os.path.join(plots_dir, "region_distribution.png")),
    }
