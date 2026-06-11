"""Data preprocessing for the Global Fuel Price Predictor.

This module exposes :class:`DataPreprocessor`, a self-contained, serialisable
feature pipeline that turns the raw ``global_fuel_prices_2020_2026.csv`` records
into a numeric design matrix suitable for the scikit-learn regressors used in
this project.

Feature engineering summary
---------------------------
* ``region``         -> One-Hot Encoding (``drop_first=True``) -> 6 columns
* ``income_level``   -> Ordinal encoding (Low=0, Middle=1, High=2)
* ``subsidy_level``  -> Ordinal encoding (Low=0, Medium=1, High=2, Very High=3)
* ``country``        -> Label encoding (84 categories, median fallback for unseen)
* ``brent_crude_usd``-> StandardScaler
* ``tax_percentage`` -> StandardScaler
* ``year``           -> extracted from ``date`` then StandardScaler
* ``month``          -> extracted from ``date`` then StandardScaler

The target variable is ``petrol_usd_liter``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class DataPreprocessor:
    """Encode raw fuel-price records into a model-ready numeric matrix.

    The preprocessor is *fitted* once on the training data (learning the region
    vocabulary, country label map and feature scaler) and can subsequently
    transform unseen records consistently -- including single rows coming from
    the Gradio web app.

    Attributes
    ----------
    region_columns : list of str
        One-hot column names kept after ``drop_first`` (sorted, first dropped).
    income_map, subsidy_map : dict
        Fixed ordinal mappings for the two ordinal categorical features.
    country_to_code : dict
        Maps each training country to an integer label.
    country_median_code : int
        Fallback code used when an unseen country is encountered at inference.
    scaler : StandardScaler
        Fitted scaler for ``[brent_crude_usd, tax_percentage, year, month]``.
    feature_names : list of str
        Ordered names of the columns in the produced design matrix.
    """

    #: Target column predicted by every model in the project.
    TARGET = "petrol_usd_liter"

    #: Ordinal mapping for ``income_level``.
    INCOME_MAP = {"Low": 0, "Middle": 1, "High": 2}

    #: Ordinal mapping for ``subsidy_level``.
    SUBSIDY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

    #: Numeric columns fed through the :class:`StandardScaler`.
    SCALED_COLS = ["brent_crude_usd", "tax_percentage", "year", "month"]

    def __init__(self) -> None:
        """Create an unfitted preprocessor."""
        self.region_categories: List[str] = []
        self.region_columns: List[str] = []
        self.income_map = dict(self.INCOME_MAP)
        self.subsidy_map = dict(self.SUBSIDY_MAP)
        self.country_to_code: dict = {}
        self.country_median_code: int = 0
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with ``year``/``month`` columns from ``date``.

        If ``year`` and ``month`` already exist (e.g. for single-row inference)
        the frame is returned untouched.
        """
        df = df.copy()
        if "year" in df.columns and "month" in df.columns:
            return df
        if "date" not in df.columns:
            raise ValueError(
                "Input frame must contain either a 'date' column or both "
                "'year' and 'month' columns."
            )
        parsed = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = parsed.dt.year
        df["month"] = parsed.dt.month
        return df

    def _encode_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode ``region`` honouring the fitted ``drop_first`` scheme."""
        out = pd.DataFrame(index=df.index)
        regions = df["region"].astype(str)
        for col in self.region_columns:
            category = col[len("region_"):]
            out[col] = (regions == category).astype(float)
        return out

    def _encode_country(self, series: pd.Series) -> np.ndarray:
        """Label-encode ``country`` with a median fallback for unseen values."""
        return series.astype(str).map(
            lambda c: self.country_to_code.get(c, self.country_median_code)
        ).astype(float).to_numpy()

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Assemble the ordered numeric design matrix from *df* (post date parse)."""
        region_part = self._encode_region(df)

        income = df["income_level"].astype(str).map(self.income_map)
        subsidy = df["subsidy_level"].astype(str).map(self.subsidy_map)
        if income.isnull().any():
            raise ValueError(f"Unknown income_level value(s): "
                             f"{df['income_level'][income.isnull()].unique()}")
        if subsidy.isnull().any():
            raise ValueError(f"Unknown subsidy_level value(s): "
                             f"{df['subsidy_level'][subsidy.isnull()].unique()}")

        country = self._encode_country(df["country"])
        scaled = self.scaler.transform(df[self.SCALED_COLS].astype(float))

        # Column order MUST match ``self.feature_names``.
        matrix = np.column_stack([
            region_part.to_numpy(),
            income.to_numpy(dtype=float),
            subsidy.to_numpy(dtype=float),
            country,
            scaled,
        ])
        return matrix

    def _compose_feature_names(self) -> List[str]:
        """Return the ordered feature names matching :meth:`_build_matrix`."""
        return (
            list(self.region_columns)
            + ["income_level", "subsidy_level", "country"]
            + list(self.SCALED_COLS)
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self, df: pd.DataFrame, test_size: float = 0.2,
        test_data_path: str = "data/test_data.pkl",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Fit the pipeline and return a train/test split of the design matrix.

        The raw frame is split *first* (so the scaler is fitted on training rows
        only, avoiding leakage), the encoders/scaler are learned on the training
        portion, then both portions are transformed.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw records containing all required columns plus ``date`` and the
            target ``petrol_usd_liter``.
        test_size : float, default 0.2
            Fraction of rows held out for testing (80:20 split).
        test_data_path : str
            Where to persist ``(X_test, y_test, feature_names)`` for the
            analysis notebook.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        logger.info("Fitting DataPreprocessor on %d records.", len(df))
        df = self._extract_date_parts(df)

        required = {
            "region", "income_level", "subsidy_level", "country",
            *self.SCALED_COLS, self.TARGET,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        y_full = df[self.TARGET].astype(float)

        # Split raw rows first to keep the scaler honest.
        df_train, df_test, y_train, y_test = train_test_split(
            df, y_full, test_size=test_size,
            random_state=RANDOM_STATE, shuffle=True,
        )
        logger.info("Train/test split: %d train / %d test rows.",
                    len(df_train), len(df_test))

        # --- Learn vocabularies / scaler on the training portion only --- #
        self.region_categories = sorted(df_train["region"].astype(str).unique())
        # drop_first=True -> drop the first (alphabetically) category.
        self.region_columns = [f"region_{c}" for c in self.region_categories[1:]]
        logger.info("Region one-hot columns (drop_first): %s", self.region_columns)

        countries = sorted(df_train["country"].astype(str).unique())
        self.country_to_code = {c: i for i, c in enumerate(countries)}
        self.country_median_code = int(np.median(list(self.country_to_code.values())))
        logger.info("Label-encoded %d countries (median fallback code=%d).",
                    len(countries), self.country_median_code)

        self.scaler = StandardScaler()
        self.scaler.fit(df_train[self.SCALED_COLS].astype(float))

        self.feature_names = self._compose_feature_names()
        self.is_fitted = True

        X_train = self._build_matrix(df_train)
        X_test = self._build_matrix(df_test)
        y_train_arr = y_train.to_numpy(dtype=float)
        y_test_arr = y_test.to_numpy(dtype=float)

        logger.info("Design matrix shape: X_train=%s, X_test=%s",
                    X_train.shape, X_test.shape)

        # Persist test split for the analysis notebook.
        if test_data_path:
            os.makedirs(os.path.dirname(test_data_path) or ".", exist_ok=True)
            joblib.dump(
                {
                    "X_test": X_test,
                    "y_test": y_test_arr,
                    "feature_names": self.feature_names,
                    "df_test": df_test.reset_index(drop=True),
                },
                test_data_path,
            )
            logger.info("Saved test data to %s", test_data_path)

        return X_train, X_test, y_train_arr, y_test_arr, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new raw records into the fitted design matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            Records with the same schema as training (``date`` or
            ``year``/``month`` accepted).

        Returns
        -------
        numpy.ndarray of shape ``(n_rows, n_features)``
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transform().")
        df = self._extract_date_parts(df)
        return self._build_matrix(df)

    def prepare_single_input(
        self, country: str, region: str, income_level: str,
        subsidy_level: str, brent_crude: float, tax_pct: float,
        year: int, month: int,
    ) -> np.ndarray:
        """Build a single ``(1, n_features)`` row for live inference.

        Unknown countries fall back to the median label code (see the class
        docstring). All arguments mirror the dataset's raw columns.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before inference.")
        row = pd.DataFrame([{
            "country": country,
            "region": region,
            "income_level": income_level,
            "subsidy_level": subsidy_level,
            "brent_crude_usd": float(brent_crude),
            "tax_percentage": float(tax_pct),
            "year": int(year),
            "month": int(month),
        }])
        return self._build_matrix(row)

    def get_feature_importance_names(self) -> List[str]:
        """Return the ordered feature names after encoding.

        Useful for labelling Random Forest feature-importance charts.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted first.")
        return list(self.feature_names)

    @property
    def n_features(self) -> int:
        """Number of columns in the produced design matrix."""
        return len(self.feature_names)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialise the fitted preprocessor to *path* via joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Saved DataPreprocessor to %s", path)

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load a previously saved preprocessor from *path*."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a DataPreprocessor.")
        logger.info("Loaded DataPreprocessor from %s", path)
        return obj
