"""Leakage-safe preprocessing for country-mode fuel-price prediction.

The dataset stores three parallel fuel-price targets. For petrol prediction,
diesel and LPG prices are deliberately excluded because using either would leak
near-identical target information. Country is the only categorical model
feature and is one-hot encoded; region, income level and subsidy level remain
country metadata for display, not independent scenario controls. Historical
country priors and trend priors are learned from training rows only so the
models can behave as forecasting models instead of treating ``year`` as a
mostly decorative scenario field.

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
    INPUT_FEATURES = ["brent_crude_usd", "tax_percentage", "year", "month"]
    NUMERIC_FEATURES = INPUT_FEATURES
    COUNTRY_PRICE_PRIOR = "country_price_prior"
    FORECAST_FEATURES = [
        "brent_crude_usd",
        "tax_percentage",
        "year",
        "month",
        "time_index",
        "forecast_horizon_months",
        "month_sin",
        "month_cos",
        "country_trend_per_month",
        "country_trend_forecast_prior",
    ]

    def __init__(self) -> None:
        self.country_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=float
        )
        self.countries: List[str] = []
        self.country_profiles: dict[str, dict[str, str]] = {}
        self.country_price_stats: dict[str, dict[str, float]] = {}
        self.country_price_priors: dict[str, float] = {}
        self.global_price_prior: float = 0.0
        self.country_trend_slopes: dict[str, float] = {}
        self.country_trend_intercepts: dict[str, float] = {}
        self.global_trend_slope: float = 0.0
        self.global_trend_intercept: float = 0.0
        self.first_train_period_index: int = 0
        self.last_train_period_index: int = 0
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
        required = {"country", *self.INPUT_FEATURES}
        if require_target:
            required.update({"date", self.TARGET, *self.COUNTRY_METADATA})
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _period_index(year: pd.Series, month: pd.Series) -> pd.Series:
        """Return a monthly integer index for forecasting features."""
        return year.astype(int) * 12 + month.astype(int)

    @staticmethod
    def _fit_linear_trend(
        period_index: np.ndarray, target: np.ndarray,
    ) -> tuple[float, float]:
        """Fit ``target = intercept + slope * time_index`` safely."""
        x = np.asarray(period_index, dtype=float)
        y = np.asarray(target, dtype=float)
        if len(np.unique(x)) < 2:
            return 0.0, float(np.mean(y)) if len(y) else 0.0
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(slope), float(intercept)

    def _fit_country_trends(self, df_train: pd.DataFrame) -> None:
        """Learn per-country linear trend priors from training rows only."""
        period_index = self._period_index(df_train["year"], df_train["month"])
        self.first_train_period_index = int(period_index.min())
        self.last_train_period_index = int(period_index.max())

        x_global = period_index.to_numpy() - self.first_train_period_index
        self.global_trend_slope, self.global_trend_intercept = (
            self._fit_linear_trend(
                x_global, df_train[self.TARGET].astype(float).to_numpy()
            )
        )

        self.country_trend_slopes = {}
        self.country_trend_intercepts = {}
        for country, group in df_train.groupby("country"):
            group_period = self._period_index(group["year"], group["month"])
            group_x = group_period.to_numpy() - self.first_train_period_index
            slope, intercept = self._fit_linear_trend(
                group_x, group[self.TARGET].astype(float).to_numpy()
            )
            self.country_trend_slopes[str(country)] = slope
            self.country_trend_intercepts[str(country)] = intercept

    def _build_forecast_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build market and time features, including train-only trend priors."""
        period_index = self._period_index(df["year"], df["month"])
        time_index = period_index.astype(float) - float(self.first_train_period_index)
        horizon = period_index.astype(float) - float(self.last_train_period_index)
        month = df["month"].astype(float)
        month_angle = 2 * np.pi * (month - 1) / 12
        country = df["country"].astype(str)
        trend_slope = (
            country.map(self.country_trend_slopes)
            .fillna(self.global_trend_slope)
            .astype(float)
        )
        trend_intercept = (
            country.map(self.country_trend_intercepts)
            .fillna(self.global_trend_intercept)
            .astype(float)
        )
        trend_prior = np.maximum(0.0, trend_intercept + trend_slope * time_index)
        features = pd.DataFrame({
            "brent_crude_usd": df["brent_crude_usd"].astype(float),
            "tax_percentage": df["tax_percentage"].astype(float),
            "year": df["year"].astype(float),
            "month": month,
            "time_index": time_index,
            "forecast_horizon_months": horizon,
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "country_trend_per_month": trend_slope,
            "country_trend_forecast_prior": trend_prior,
        })
        return features[self.FORECAST_FEATURES].to_numpy(dtype=float)

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Encode country and append unscaled forecasting features."""
        countries = self.country_encoder.transform(df[["country"]].astype(str))
        country_prior = (
            df["country"].astype(str)
            .map(self.country_price_priors)
            .fillna(self.global_price_prior)
            .astype(float)
            .to_numpy()
        )
        numeric = self._build_forecast_features(df)
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
        self._fit_country_trends(df_train)
        country_features = self.country_encoder.get_feature_names_out(["country"])
        self.feature_names = (
            list(country_features)
            + [self.COUNTRY_PRICE_PRIOR]
            + list(self.FORECAST_FEATURES)
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
