"""Training pipeline for the Global Fuel Price Predictor.

Loads the dataset, fits the :class:`DataPreprocessor`, trains the requested
model(s), evaluates them on a shared held-out test set, writes a comparison
table (``data/model_comparison.json``) and a comparison chart
(``data/plots/model_comparison.png``), prints a tidy summary and persists every
artifact under ``models/``.

Usage
-----
    python train.py                # train all three models (default)
    python train.py --model knn    # train only K-Nearest Neighbours
    python train.py --model svm    # train only Support Vector Regression
    python train.py --model rf     # train only Random Forest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Make the local ``src`` package importable regardless of the caller's CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import DataPreprocessor  # noqa: E402
from src.models import KNNModel, RandomForestModel, SVMModel  # noqa: E402
from src.models._common import (  # noqa: E402
    ACTION_BLUE,
    CORAL,
    DEEP_GREEN,
    PLOT_DPI,
    PRIMARY,
    _apply_style,
)
from src.eda import generate_overview_plots  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #
DATA_PATH = os.path.join(BASE_DIR, "data", "global_fuel_prices_2020_2026.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.pkl")
COMPARISON_JSON = os.path.join(DATA_DIR, "model_comparison.json")
COMPARISON_PLOT = os.path.join(PLOTS_DIR, "model_comparison.png")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

MODEL_FILES = {
    "KNN": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "rf_model.pkl"),
}
MODEL_KEY_TO_NAME = {"knn": "KNN", "svm": "SVM", "rf": "Random Forest"}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_dirs() -> None:
    """Create the output directories if they do not yet exist."""
    for d in (MODELS_DIR, DATA_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)


def _load_dataset() -> pd.DataFrame:
    """Load and lightly validate the raw CSV dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded dataset: %d rows x %d columns from %s",
                df.shape[0], df.shape[1], os.path.basename(DATA_PATH))
    return df


def _build_model(name: str):
    """Instantiate a fresh model object for *name* (display name)."""
    return {
        "KNN": KNNModel,
        "SVM": SVMModel,
        "Random Forest": RandomForestModel,
    }[name]()


def _train_one(name: str, model, X_train, y_train, feature_names=None) -> float:
    """Train *model* and return wall-clock training seconds."""
    logger.info("=" * 64)
    logger.info("Training %s ...", name)
    start = time.perf_counter()
    if name == "SVM":
        model.train(X_train, y_train, subsample=True, subsample_size=10_000)
    elif name == "KNN":
        # KNN benefits from amplifying the dominant ``country`` column so that
        # neighbours are matched within the same country (see KNNModel.train).
        country_index = (feature_names.index("country")
                         if feature_names and "country" in feature_names
                         else None)
        model.train(X_train, y_train, country_index=country_index)
    else:
        model.train(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("%s training finished in %.1fs", name, elapsed)
    return elapsed


def _load_comparison() -> dict:
    """Load the existing comparison JSON, or return a fresh skeleton."""
    if os.path.exists(COMPARISON_JSON):
        try:
            with open(COMPARISON_JSON, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "models" in data:
                return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read existing %s; starting fresh.",
                           COMPARISON_JSON)
    return {"generated_at": None, "models": {}, "best_model": None,
            "feature_names": []}


def _recompute_best(comparison: dict) -> None:
    """Set ``best_model`` to the entry with the lowest test RMSE."""
    best_name, best_rmse = None, float("inf")
    for name, entry in comparison["models"].items():
        rmse = entry.get("metrics", {}).get("RMSE")
        if rmse is not None and rmse < best_rmse:
            best_name, best_rmse = name, rmse
    if best_name is not None:
        comparison["best_model"] = {
            "name": best_name, "by": "RMSE", "RMSE": round(best_rmse, 5)}


def _plot_comparison(comparison: dict) -> None:
    """Render the 4-panel (MAE, RMSE, R², MAPE) model-comparison bar chart."""
    models = list(comparison["models"].keys())
    if not models:
        return
    plt = _apply_style()
    colours = [DEEP_GREEN, CORAL, ACTION_BLUE, PRIMARY][: len(models)]

    panels = [
        ("MAE", "MAE (USD/L) — lower is better", False),
        ("RMSE", "RMSE (USD/L) — lower is better", False),
        ("R2", "R² — higher is better", True),
        ("MAPE", "MAPE (%) — lower is better", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16,
                 fontweight="bold", color=PRIMARY)

    for ax, (metric, title, higher_better) in zip(axes.ravel(), panels):
        values = [comparison["models"][m]["metrics"][metric] for m in models]
        bars = ax.bar(models, values, color=colours, edgecolor=PRIMARY,
                      linewidth=0.8, width=0.6)
        # Highlight the best bar for this metric.
        best_idx = (int(np.argmax(values)) if higher_better
                    else int(np.argmin(values)))
        bars[best_idx].set_edgecolor(CORAL)
        bars[best_idx].set_linewidth(2.5)
        ax.set_title(title, color=PRIMARY, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, color=PRIMARY)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=PRIMARY)
        ax.margins(y=0.15)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(COMPARISON_PLOT) or ".", exist_ok=True)
    fig.savefig(COMPARISON_PLOT, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison chart to %s", COMPARISON_PLOT)


def _print_summary(comparison: dict) -> None:
    """Print a fixed-width comparison table to the terminal."""
    models = list(comparison["models"].keys())
    header = (f"{'Model':<16}{'MAE':>10}{'RMSE':>10}{'R2':>10}"
              f"{'MAPE %':>12}{'Akurasi%':>11}")
    line = "-" * len(header)
    print("\n" + line)
    print("MODEL COMPARISON SUMMARY".center(len(header)))
    print(line)
    print(header)
    print(line)
    for m in models:
        mt = comparison["models"][m]["metrics"]
        # Akurasi% = R² x 100 (proportion of price variance explained).
        print(f"{m:<16}{mt['MAE']:>10.4f}{mt['RMSE']:>10.4f}"
              f"{mt['R2']:>10.4f}{mt['MAPE']:>12.2f}{mt['R2'] * 100:>11.2f}")
    print(line)
    if comparison.get("best_model"):
        best = comparison["best_model"]
        print(f"BEST MODEL (lowest RMSE): {best['name']} "
              f"(RMSE = {best['RMSE']:.4f})")
    print(line + "\n")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(selected: str = "all") -> dict:
    """Execute the training pipeline for *selected* (``all|knn|svm|rf``)."""
    _ensure_dirs()
    df = _load_dataset()

    # 1) Preprocess (fit + 80/20 split, persists test_data.pkl).
    logger.info("Fitting preprocessor and building train/test split...")
    pre = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = pre.fit_transform(
        df, test_data_path=TEST_DATA_PATH)
    pre.save(PREPROCESSOR_PATH)

    # 2) Decide which models to train.
    if selected == "all":
        to_train = ["KNN", "SVM", "Random Forest"]
    else:
        to_train = [MODEL_KEY_TO_NAME[selected]]
    logger.info("Models to train: %s", to_train)

    comparison = _load_comparison()
    comparison["feature_names"] = feature_names

    # 3) Train + evaluate each model.
    for name in to_train:
        model = _build_model(name)
        elapsed = _train_one(name, model, X_train, y_train, feature_names)
        metrics = model.evaluate(X_test, y_test)

        # Diagnostic plots.
        key = name.lower().replace(" ", "_")
        model.plot_predictions_vs_actual(
            X_test, y_test, os.path.join(PLOTS_DIR, f"{key}_pred_vs_actual.png"))
        model.plot_residuals(
            X_test, y_test, os.path.join(PLOTS_DIR, f"{key}_residuals.png"))
        if isinstance(model, RandomForestModel):
            model.plot_feature_importance(
                feature_names, os.path.join(PLOTS_DIR, "rf_feature_importance.png"))

        # Persist model + record entry.
        model.save(MODEL_FILES[name])
        entry = {
            "metrics": {k: round(v, 5) for k, v in metrics.items()},
            "best_params": model.get_best_params(),
            "train_seconds": round(elapsed, 2),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        if isinstance(model, SVMModel):
            entry["subsample_info"] = model.subsample_info
        comparison["models"][name] = entry

    # 4) Finalise comparison artefacts.
    comparison["generated_at"] = datetime.now().isoformat(timespec="seconds")
    _recompute_best(comparison)
    with open(COMPARISON_JSON, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    logger.info("Saved comparison table to %s", COMPARISON_JSON)

    _plot_comparison(comparison)

    # 5) Dataset-overview plots (used by the web app).
    generate_overview_plots(df, PLOTS_DIR)

    # 6) Terminal summary.
    _print_summary(comparison)
    return comparison


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fuel-price prediction models.")
    parser.add_argument(
        "--model", choices=["knn", "svm", "rf"], default=None,
        help="Train a single model; omit to train all three (default).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    selected = args.model if args.model else "all"
    logger.info("Starting training pipeline (selection=%s)", selected)
    run(selected)
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
