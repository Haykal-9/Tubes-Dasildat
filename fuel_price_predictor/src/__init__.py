"""Source package for the Global Fuel Price Predictor project.

Exposes the :class:`DataPreprocessor` used across the training pipeline,
the inference web app and the analysis notebook.
"""

from .preprocessing import DataPreprocessor

__all__ = ["DataPreprocessor"]
