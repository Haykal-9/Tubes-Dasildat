"""Start-up guard that guarantees trained artifacts exist before the app runs.

On Hugging Face Spaces the repository may be cloned without pre-trained models.
``ensure_models_ready()`` checks for the preprocessor and the three model files
and, if any are missing, runs the full training pipeline once. ``app.py`` calls
this at import time.
"""

from __future__ import annotations

import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

REQUIRED_ARTIFACTS = [
    os.path.join(MODELS_DIR, "preprocessor.pkl"),
    os.path.join(MODELS_DIR, "knn_model.pkl"),
    os.path.join(MODELS_DIR, "svm_model.pkl"),
    os.path.join(MODELS_DIR, "rf_model.pkl"),
]
COMPARISON_JSON = os.path.join(DATA_DIR, "model_comparison.json")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app_startup")


def artifacts_present() -> bool:
    """Return True when every required model artifact is on disk."""
    return all(os.path.exists(p) for p in REQUIRED_ARTIFACTS)


def overview_plots_present() -> bool:
    """Return True when the dataset-overview charts already exist."""
    expected = [
        "petrol_distribution.png",
        "price_timeseries.png",
        "region_distribution.png",
        "model_comparison.png",
    ]
    return all(os.path.exists(os.path.join(PLOTS_DIR, p)) for p in expected)


def ensure_models_ready(force: bool = False) -> bool:
    """Make sure trained models and plots exist, training if necessary.

    Parameters
    ----------
    force : bool, default False
        Retrain even if artifacts already exist.

    Returns
    -------
    bool
        ``True`` if artifacts are ready after the call, ``False`` if training
        was attempted but failed.
    """
    if artifacts_present() and not force:
        logger.info("All model artifacts present — skipping training.")
        if not overview_plots_present():
            _regenerate_overview_plots()
        return True

    logger.warning("Model artifacts missing — launching training pipeline. "
                   "This can take several minutes on first start-up...")
    try:
        import train  # local module
        train.run("all")
        logger.info("Training pipeline completed during start-up.")
        return artifacts_present()
    except Exception:  # noqa: BLE001 - surface any failure to the caller/logs
        logger.exception("Automatic training failed during start-up.")
        return False


def _regenerate_overview_plots() -> None:
    """Re-create the (git-ignored) dataset-overview plots from the CSV."""
    try:
        import pandas as pd

        from src.eda import generate_overview_plots

        csv = os.path.join(DATA_DIR, "global_fuel_prices_2020_2026.csv")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        generate_overview_plots(pd.read_csv(csv), PLOTS_DIR)
        logger.info("Regenerated dataset-overview plots.")
    except Exception:  # noqa: BLE001
        logger.exception("Could not regenerate overview plots.")


if __name__ == "__main__":
    ok = ensure_models_ready()
    raise SystemExit(0 if ok else 1)
