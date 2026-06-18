"""Leakage-safe preprocessing for country-mode fuel-price prediction.

The dataset stores three parallel fuel-price targets. For petrol prediction,
diesel and LPG prices are deliberately excluded because using either would leak
near-identical target information. Country is the only categorical model
feature and is one-hot encoded; region, income level and subsidy level remain
country metadata for display, not independent scenario controls. A historical
country-price prior is learned from training rows only so tree models retain
the selected country's price level when market inputs are out of distribution.

The split is explicitly chronological: all rows before the latest year are
training data and the latest year is held out for testing. Numeric features are
left unscaled here because KNN and SVM own their StandardScaler inside their
serialised sklearn Pipeline. Random Forest consumes the same raw matrix.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Build one-hot country plus raw market/time features."""

    TARGET = "petrol_usd_liter"
    PARALLEL_TARGETS = ["petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter"]
    COUNTRY_METADATA = ["region", "income_level", "subsidy_level"]
    NUMERIC_FEATURES = ["brent_crude_usd", "tax_percentage", "year", "month"]
    COUNTRY_PRICE_PRIOR = "country_price_prior"

    def __init__(self) -> None:
        self.country_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=float
        )
        self.countries: List[str] = []
        self.country_profiles: dict[str, dict[str, str]] = {}
        self.country_price_stats: dict[str, dict[str, float]] = {}
        self.country_price_priors: dict[str, float] = {}
        self.global_price_prior: float = 0.0
        self.country_tax_ranges: dict[str, tuple[float, float]] = {}
        self.brent_year_ranges: dict[int, tuple[float, float]] = {}
        self.feature_names: List[str] = []
        self.train_years: List[int] = []
        self.test_year: int | None = None
        self.is_fitted = False

    @staticmethod
    def _extract_date_parts(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with numeric year and month columns."""
        df = df.copy()
        if "year" in df.columns and "month" in df.columns:
            return df
        if "date" not in df.columns:
            raise ValueError(
                "Input must contain either 'date' or both 'year' and 'month'."
            )
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if parsed.isnull().any():
            raise ValueError("The 'date' column contains invalid values.")
        df["year"] = parsed.dt.year.astype(int)
        df["month"] = parsed.dt.month.astype(int)
        return df

    def _validate_columns(self, df: pd.DataFrame, require_target: bool) -> None:
        required = {"country", *self.NUMERIC_FEATURES}
        if require_target:
            required.update({"date", self.TARGET, *self.COUNTRY_METADATA})
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Encode country and append unscaled numeric features."""
        countries = self.country_encoder.transform(df[["country"]].astype(str))
        country_prior = (
            df["country"].astype(str)
            .map(self.country_price_priors)
            .fillna(self.global_price_prior)
            .astype(float)
            .to_numpy()
        )
        numeric = df[self.NUMERIC_FEATURES].astype(float).to_numpy()
        return np.column_stack([countries, country_prior, numeric])

    def _fit_metadata(self, df: pd.DataFrame) -> None:
        profiles = (
            df.groupby("country")[self.COUNTRY_METADATA]
            .first()
            .astype(str)
        )
        self.country_profiles = profiles.to_dict(orient="index")

        stats = df.groupby("country")[self.TARGET].agg(
            ["min", "mean", "max", "std", "count"]
        )
        self.country_price_stats = {
            str(country): {
                key: float(value) for key, value in row.items()
            }
            for country, row in stats.iterrows()
        }

        tax_ranges = df.groupby("country")["tax_percentage"].agg(["min", "max"])
        self.country_tax_ranges = {
            str(country): (float(row["min"]), float(row["max"]))
            for country, row in tax_ranges.iterrows()
        }

        ranges = df.groupby("year")["brent_crude_usd"].agg(["min", "max"])
        self.brent_year_ranges = {
            int(year): (float(row["min"]), float(row["max"]))
            for year, row in ranges.iterrows()
        }

    def fit_transform(
        self, df: pd.DataFrame, test_size: float = 0.2,
        test_data_path: str = "data/test_data.pkl",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Fit on all pre-latest-year rows and hold out the latest year."""
        del test_size  # The brief requires an explicit year-based holdout.
        df = self._extract_date_parts(df)
        self._validate_columns(df, require_target=True)

        # Guard against accidental target leakage in future edits.
        leaked = set(self.PARALLEL_TARGETS[1:]) & set(self.NUMERIC_FEATURES)
        if leaked:
            raise RuntimeError(f"Parallel fuel targets leaked into features: {leaked}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df = df.sort_values(["date", "country"]).reset_index(drop=True)

        self.test_year = int(df["year"].max())
        self.train_years = sorted(
            int(year) for year in df.loc[df["year"] < self.test_year, "year"].unique()
        )
        df_train = df[df["year"] < self.test_year].copy()
        df_test = df[df["year"] == self.test_year].copy()
        if df_train.empty or df_test.empty:
            raise ValueError("Year-based split produced an empty train or test set.")
        logger.info(
            "Year holdout: train=%s (%d rows), test=%d (%d rows).",
            f"{self.train_years[0]}-{self.train_years[-1]}",
            len(df_train), self.test_year, len(df_test),
        )

        self.country_encoder.fit(df_train[["country"]].astype(str))
        self.countries = sorted(df_train["country"].astype(str).unique().tolist())
        self.country_price_priors = {
            str(country): float(value)
            for country, value in
            df_train.groupby("country")[self.TARGET].mean().items()
        }
        self.global_price_prior = float(df_train[self.TARGET].mean())
        country_features = self.country_encoder.get_feature_names_out(["country"])
        self.feature_names = (
            list(country_features)
            + [self.COUNTRY_PRICE_PRIOR]
            + list(self.NUMERIC_FEATURES)
        )
        self._fit_metadata(df)
        self.is_fitted = True

        X_train = self._build_matrix(df_train)
        X_test = self._build_matrix(df_test)
        y_train = df_train[self.TARGET].astype(float).to_numpy()
        y_test = df_test[self.TARGET].astype(float).to_numpy()

        if test_data_path:
            os.makedirs(os.path.dirname(test_data_path) or ".", exist_ok=True)
            joblib.dump(
                {
                    "X_test": X_test,
                    "y_test": y_test,
                    "feature_names": self.feature_names,
                    "df_test": df_test.reset_index(drop=True),
                    "split_type": "year_holdout",
                    "train_years": self.train_years,
                    "test_year": self.test_year,
                },
                test_data_path,
            )
        return X_train, X_test, y_train, y_test, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform records using country plus market/time fields only."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transform().")
        df = self._extract_date_parts(df)
        self._validate_columns(df, require_target=False)
        return self._build_matrix(df)

    def prepare_single_input(
        self, country: str, region: str | None, income_level: str | None,
        subsidy_level: str | None, brent_crude: float, tax_pct: float,
        year: int, month: int,
    ) -> np.ndarray:
        """Build one inference row; locked country metadata is intentionally ignored."""
        del region, income_level, subsidy_level
        row = pd.DataFrame([{
            "country": country,
            "brent_crude_usd": float(brent_crude),
            "tax_percentage": float(tax_pct),
            "year": int(year),
            "month": int(month),
        }])
        return self.transform(row)

    def get_country_profile(self, country: str) -> dict[str, str]:
        """Return locked metadata for a known country."""
        return dict(self.country_profiles.get(str(country), {}))

    def get_country_price_stats(self, country: str) -> dict[str, float]:
        """Return historical petrol-price statistics for a known country."""
        return dict(self.country_price_stats.get(str(country), {}))

    def get_country_tax_range(self, country: str) -> tuple[float, float] | None:
        """Return the observed tax range for a country, if available."""
        return self.country_tax_ranges.get(str(country))

    def get_brent_range(self, year: int) -> tuple[float, float] | None:
        """Return the observed Brent range for a year, if available."""
        return self.brent_year_ranges.get(int(year))

    def get_feature_importance_names(self) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted first.")
        return list(self.feature_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Saved DataPreprocessor to %s", path)

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a DataPreprocessor.")
        logger.info("Loaded DataPreprocessor from %s", path)
        return obj
