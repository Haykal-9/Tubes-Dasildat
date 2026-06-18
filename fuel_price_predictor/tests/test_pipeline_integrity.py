"""Regression tests for the leakage-safe country-mode pipeline."""

from __future__ import annotations

import os
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models import KNNModel, RandomForestModel, SVMModel  # noqa: E402
from src.preprocessing import DataPreprocessor  # noqa: E402


class PipelineIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        models_dir = os.path.join(BASE_DIR, "models")
        cls.pre = DataPreprocessor.load(
            os.path.join(models_dir, "preprocessor.pkl"))
        cls.models = {
            "KNN": KNNModel.load(os.path.join(models_dir, "knn_model.pkl")),
            "SVM": SVMModel.load(os.path.join(models_dir, "svm_model.pkl")),
            "Random Forest": RandomForestModel.load(
                os.path.join(models_dir, "rf_model.pkl")),
        }

    def _input(self, **changes) -> np.ndarray:
        values = {
            "country": "Algeria",
            "region": "Africa",
            "income_level": "Middle",
            "subsidy_level": "Very High",
            "brent_crude": 124,
            "tax_pct": 15,
            "year": 2026,
            "month": 3,
        }
        values.update(changes)
        return self.pre.prepare_single_input(**values)

    def test_parallel_fuel_targets_are_not_features(self) -> None:
        forbidden = {"petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter"}
        self.assertFalse(forbidden & set(self.pre.feature_names))

    def test_country_price_prior_uses_training_years_only(self) -> None:
        df = pd.read_csv(os.path.join(
            BASE_DIR, "data", "global_fuel_prices_2020_2026.csv"))
        dates = pd.to_datetime(df["date"])
        train = df[dates.dt.year < self.pre.test_year]
        expected = train.groupby("country")["petrol_usd_liter"].mean()
        self.assertIn(self.pre.COUNTRY_PRICE_PRIOR, self.pre.feature_names)
        self.assertAlmostEqual(
            self.pre.country_price_priors["Indonesia"],
            float(expected["Indonesia"]),
        )

    def test_latest_year_is_the_only_test_year(self) -> None:
        payload = joblib.load(os.path.join(BASE_DIR, "data", "test_data.pkl"))
        years = pd.to_datetime(payload["df_test"]["date"]).dt.year.unique()
        self.assertEqual(years.tolist(), [2026])
        self.assertEqual(payload["split_type"], "year_holdout")

    def test_country_metadata_is_locked_and_ignored_by_model_input(self) -> None:
        valid = self._input()
        impossible = self._input(
            region="Europe", income_level="High", subsidy_level="Low")
        np.testing.assert_array_equal(valid, impossible)

    def test_knn_and_svm_persist_scaler_pipeline(self) -> None:
        for name in ("KNN", "SVM"):
            model = self.models[name].model
            with self.subTest(model=name):
                self.assertIsInstance(model, Pipeline)
                self.assertIsInstance(model.named_steps["scale"], StandardScaler)
        self.assertNotIsInstance(self.models["Random Forest"].model, Pipeline)

    def test_predictions_are_finite_and_country_sensitive(self) -> None:
        for name, model in self.models.items():
            algeria = float(model.predict(self._input(country="Algeria"))[0])
            norway = float(model.predict(self._input(country="Norway"))[0])
            with self.subTest(model=name):
                self.assertTrue(np.isfinite([algeria, norway]).all())
                self.assertGreater(algeria, 0)
                self.assertGreater(norway, 0)
                self.assertGreater(abs(algeria - norway), 0.1)

    def test_random_in_range_scenarios_are_non_negative(self) -> None:
        rng = np.random.RandomState(42)
        rows = []
        years = list(self.pre.brent_year_ranges)
        for _ in range(250):
            year = int(rng.choice(years))
            low, high = self.pre.brent_year_ranges[year]
            rows.append({
                "country": rng.choice(self.pre.countries),
                "brent_crude_usd": rng.uniform(low, high),
                "tax_percentage": rng.uniform(0, 100),
                "year": year,
                "month": rng.randint(1, 13),
            })
        X = self.pre.transform(pd.DataFrame(rows))
        for name, model in self.models.items():
            predictions = model.predict(X)
            with self.subTest(model=name):
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())

    def test_rf_retains_indonesia_price_level_for_high_tax_scenario(self) -> None:
        prediction = float(self.models["Random Forest"].predict(self._input(
            country="Indonesia",
            region="Asia",
            income_level="Middle",
            subsidy_level="High",
            brent_crude=122,
            tax_pct=36,
            year=2026,
            month=6,
        ))[0])
        stats = self.pre.get_country_price_stats("Indonesia")
        self.assertGreaterEqual(prediction, stats["min"])
        self.assertLessEqual(prediction, stats["max"])
        self.assertEqual(self.pre.get_country_tax_range("Indonesia"), (10.0, 30.0))


if __name__ == "__main__":
    unittest.main()
