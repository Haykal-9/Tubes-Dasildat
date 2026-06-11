"""Model package for the Global Fuel Price Predictor project.

Bundles the three regression estimators used in the comparison study:

* :class:`KNNModel`            -- K-Nearest Neighbours regressor (GridSearchCV)
* :class:`SVMModel`            -- Support Vector regressor (GridSearchCV, subsampled)
* :class:`RandomForestModel`   -- Random Forest regressor (RandomizedSearchCV)
"""

from .knn_model import KNNModel
from .svm_model import SVMModel
from .rf_model import RandomForestModel

__all__ = ["KNNModel", "SVMModel", "RandomForestModel"]
