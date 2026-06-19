"""Gradio web app for the Global Fuel Price Predictor.

Deployed to Hugging Face Spaces. Provides three tabs:

1. **Prediksi Harga BBM** -- interactive single-record petrol-price prediction.
2. **Perbandingan Model**  -- KNN / SVM / Random Forest metric comparison.
3. **Dataset Overview**    -- descriptive statistics and EDA charts.

Models and the preprocessor are loaded once at start-up (with a guard that
trains them automatically if the artifacts are missing). The UI applies a
minimalist design system via custom CSS.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os

import gradio as gr
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / logging
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYED_ASSET_DIR = os.path.join(BASE_DIR, "asset")
LOCAL_ASSET_DIR = os.path.join(os.path.dirname(BASE_DIR), "asset")
ASSET_DIR = (
    DEPLOYED_ASSET_DIR
    if os.path.isdir(DEPLOYED_ASSET_DIR)
    else LOCAL_ASSET_DIR
)
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "global_fuel_prices_2020_2026.csv")
COMPARISON_JSON = os.path.join(DATA_DIR, "model_comparison.json")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# Make ``src`` importable and ensure artifacts exist before anything else.
import sys  # noqa: E402

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app_startup import ensure_models_ready  # noqa: E402
from src.preprocessing import DataPreprocessor  # noqa: E402
from src.models import KNNModel, RandomForestModel, SVMModel  # noqa: E402
from src.eda import descriptive_stats  # noqa: E402

# --------------------------------------------------------------------------- #
# Design system tokens — Mintlify-inspired product dashboard
# --------------------------------------------------------------------------- #
BG = "#F7F7F7"
SURFACE = "#FFFFFF"
BORDER = "#E5E5E5"
HEADER_BG = "#0A0A0A"
TEXT_PRIMARY = "#0A0A0A"
TEXT_SECONDARY = "#5A5A5C"
TEXT_MUTED = "#888888"
ACCENT = "#00D4A4"
ACCENT_LIGHT = "#E8FBF6"
ACCENT_DARK = "#00B48A"
SECONDARY = "#3772CF"
SECONDARY_LIGHT = "#EEF4FF"
SUCCESS = "#1BA673"
SUCCESS_LIGHT = "#E8F8F2"
WARNING = "#C37D0D"
ERROR = "#D45656"

MONTHS = [
    ("Januari (01)", 1), ("Februari (02)", 2), ("Maret (03)", 3),
    ("April (04)", 4), ("Mei (05)", 5), ("Juni (06)", 6),
    ("Juli (07)", 7), ("Agustus (08)", 8), ("September (09)", 9),
    ("Oktober (10)", 10), ("November (11)", 11), ("Desember (12)", 12),
]
MODEL_DISPLAY = ["KNN", "SVM", "Random Forest"]
FUTURE_YEARS_AHEAD = 4
FUTURE_RECOMMENDED_MODEL = "SVM"

# Gradio 6 moved `css`/`theme` from the Blocks constructor to launch(); Gradio
# 4.x (the Hugging Face deployment target) requires them on Blocks. Detect the
# major version so the design-system CSS applies correctly on both.
try:
    GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except (ValueError, AttributeError):
    GRADIO_MAJOR = 4


# --------------------------------------------------------------------------- #
# Artifact loading (cached at module import)
# --------------------------------------------------------------------------- #
class AppState:
    """Container caching everything the UI needs after a single load."""

    def __init__(self) -> None:
        self.ready = False
        self.load_error = ""
        self.preprocessor: DataPreprocessor | None = None
        self.models: dict = {}
        self.comparison: dict = {}
        self.df: pd.DataFrame | None = None
        self.countries: list = []
        self.regions: list = []
        self.country_profiles: dict = {}
        self.country_price_stats: dict = {}
        self.country_tax_ranges: dict = {}
        self.brent_year_ranges: dict = {}
        self.year_min: int = 2020
        self.year_max: int = 2026
        self.forecast_year_max: int = 2030

    def model_r2(self, display_name: str) -> float | None:
        """Return the cached test R² for *display_name* (or None)."""
        entry = self.comparison.get("models", {}).get(display_name, {})
        return entry.get("metrics", {}).get("R2")


STATE = AppState()


def _load_everything() -> None:
    """Load dataset, preprocessor, models and comparison table into STATE."""
    try:
        ensure_models_ready()  # train on first run if needed

        STATE.df = pd.read_csv(CSV_PATH)
        STATE.countries = sorted(STATE.df["country"].astype(str).unique().tolist())
        STATE.regions = sorted(STATE.df["region"].astype(str).unique().tolist())
        STATE.preprocessor = DataPreprocessor.load(
            os.path.join(MODELS_DIR, "preprocessor.pkl"))
        STATE.country_profiles = STATE.preprocessor.country_profiles
        STATE.country_price_stats = STATE.preprocessor.country_price_stats
        STATE.country_tax_ranges = STATE.preprocessor.country_tax_ranges
        STATE.brent_year_ranges = STATE.preprocessor.brent_year_ranges
        STATE.year_min = min(STATE.preprocessor.train_years)
        STATE.year_max = int(STATE.preprocessor.test_year)
        STATE.forecast_year_max = STATE.year_max + FUTURE_YEARS_AHEAD
        STATE.models = {
            "KNN": KNNModel.load(os.path.join(MODELS_DIR, "knn_model.pkl")),
            "SVM": SVMModel.load(os.path.join(MODELS_DIR, "svm_model.pkl")),
            "Random Forest": RandomForestModel.load(
                os.path.join(MODELS_DIR, "rf_model.pkl")),
        }

        if os.path.exists(COMPARISON_JSON):
            with open(COMPARISON_JSON, "r", encoding="utf-8") as fh:
                STATE.comparison = json.load(fh)

        STATE.ready = True
        logger.info("App state loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        STATE.load_error = str(exc)
        logger.exception("Failed to load app artifacts.")


_load_everything()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _img(filename: str):
    """Return a plot path if it exists, else ``None`` (graceful gr.Image)."""
    path = os.path.join(PLOTS_DIR, filename)
    return path if os.path.exists(path) else None


def _asset_data_uri(filename: str) -> str:
    """Return an asset as a data URI so custom HTML can render it reliably."""
    path = os.path.join(ASSET_DIR, filename)
    try:
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as asset_file:
            encoded = base64.b64encode(asset_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except OSError:
        logger.warning("Asset not found: %s", path)
        return ""


BRAND_LOGO_DATA_URI = _asset_data_uri(
    "fuelpredict-logo-horizontal-transparent.png"
)


def _wape_colour(wape: float | None) -> str:
    """Return a display colour for lower-is-better WAPE."""
    if wape is None:
        return TEXT_MUTED
    if wape < 5:
        return SUCCESS
    if wape < 10:
        return WARNING
    return ERROR


def model_accuracy(display_name: str) -> tuple[float | None, float | None]:
    """Return temporal-test R² percentage and WAPE percentage."""
    m = STATE.comparison.get("models", {}).get(display_name, {}).get("metrics", {})
    r2, wape = m.get("R2"), m.get("WAPE")
    return (round(r2 * 100, 2) if r2 is not None else None,
            round(wape, 2) if wape is not None else None)


def country_profile_values(country: str) -> tuple[str, str, str]:
    """Return locked region, income and subsidy metadata for a country."""
    profile = STATE.country_profiles.get(str(country), {})
    return (
        profile.get("region", "Tidak diketahui"),
        profile.get("income_level", "Tidak diketahui"),
        profile.get("subsidy_level", "Tidak diketahui"),
    )


def observed_brent_range() -> tuple[float, float] | None:
    """Return the full observed Brent range across all dataset years."""
    ranges = list(STATE.brent_year_ranges.values())
    if not ranges:
        return None
    return min(low for low, _ in ranges), max(high for _, high in ranges)


def recommended_model_for_year(year: int) -> str:
    """Return the preferred model for a scenario year."""
    if int(year) > STATE.year_max:
        return FUTURE_RECOMMENDED_MODEL
    return STATE.comparison.get("best_model", {}).get("name", FUTURE_RECOMMENDED_MODEL)


def _error_card(message: str) -> str:
    """Return an HTML error card used when models are unavailable."""
    return (
        f"<div class='notice notice-error'>"
        f"<div class='notice-title'>Model belum siap</div>"
        f"<p>{message}</p>"
        f"<p>Jalankan <code>python train.py</code> "
        f"terlebih dahulu.</p></div>"
    )


def _section_label(title: str, description: str) -> str:
    """Return a compact form-section heading."""
    return (
        "<div class='form-section'>"
        f"<span>{title}</span><small>{description}</small>"
        "</div>"
    )


def predict_price(country, region, income_level, subsidy_level,
                  brent_crude, tax_pct, year, month, model_choice):
    """Predict petrol price and build the styled result panel + comparison table.

    Returns
    -------
    (html, dataframe)
        The result panel (HTML) and a 3-row comparison DataFrame.
    """
    if not STATE.ready:
        return _error_card(STATE.load_error or "Artifacts gagal dimuat."), \
            pd.DataFrame(columns=["Skenario", "Harga (USD/L)"])

    try:
        model = STATE.models[model_choice]
        # Country metadata is locked by the dataset. Ignore stale or manipulated
        # UI values so impossible country/category combinations never reach a model.
        region, income_level, subsidy_level = country_profile_values(country)
        X = STATE.preprocessor.prepare_single_input(
            country=country, region=region, income_level=income_level,
            subsidy_level=subsidy_level, brent_crude=brent_crude,
            tax_pct=tax_pct, year=int(year), month=int(month))
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # price cannot be negative
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed.")
        return _error_card(f"Prediksi gagal: {exc}"), \
            pd.DataFrame(columns=["Skenario", "Harga (USD/L)"])

    r2 = STATE.model_r2(model_choice)
    acc_r2, wape = model_accuracy(model_choice)
    wape_colour = _wape_colour(wape)
    acc_r2_text = f"{acc_r2:.2f}%" if acc_r2 is not None else "n/a"
    wape_text = f"{wape:.1f}%" if wape is not None else "n/a"
    r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
    stats = STATE.country_price_stats.get(country, {})
    tax_range = STATE.country_tax_ranges.get(country)
    scenario_year = int(year)
    is_future_year = scenario_year > STATE.year_max
    brent_range = STATE.brent_year_ranges.get(scenario_year)
    full_brent_range = observed_brent_range()
    warnings = []
    if is_future_year:
        warnings.append(
            f"Tahun {scenario_year} berada setelah data terakhir "
            f"({STATE.year_max}). Hasil ini adalah prediksi masa depan berbasis "
            "asumsi Brent dan pajak, bukan akurasi yang sudah tervalidasi.")
        if (
            full_brent_range
            and not full_brent_range[0] <= float(brent_crude) <= full_brent_range[1]
        ):
            warnings.append(
                f"Brent {float(brent_crude):.1f} berada di luar rentang "
                f"observasi dataset "
                f"({full_brent_range[0]:.1f}-{full_brent_range[1]:.1f}).")
        if model_choice != FUTURE_RECOMMENDED_MODEL:
            warnings.append(
                f"Untuk tahun masa depan, {FUTURE_RECOMMENDED_MODEL} "
                "direkomendasikan karena lebih stabil untuk prediksi masa depan.")
        if model_choice == "KNN":
            warnings.append(
                "KNN berbasis tetangga historis, jadi prediksi masa depan bisa "
                "cenderung datar setelah melewati rentang data.")
    elif brent_range is None:
        warnings.append(
            f"Tahun {int(year)} tidak tersedia dalam data training/test.")
    elif not brent_range[0] <= float(brent_crude) <= brent_range[1]:
        warnings.append(
            f"Brent {float(brent_crude):.1f} berada di luar rentang historis "
            f"{int(year)} ({brent_range[0]:.1f}–{brent_range[1]:.1f}).")
    if tax_range and not tax_range[0] <= float(tax_pct) <= tax_range[1]:
        warnings.append(
            f"Pajak {float(tax_pct):.1f}% berada di luar rentang historis "
            f"{country} ({tax_range[0]:.1f}%–{tax_range[1]:.1f}%).")
    if stats and not stats["min"] <= pred <= stats["max"]:
        warnings.append(
            f"Prediksi berada di luar rentang historis {country} "
            f"(${stats['min']:.3f}–${stats['max']:.3f}/L).")
    warning_html = "".join(
        f"<div class='notice'><strong>Perhatian:</strong> {message}</div>"
        for message in warnings
    )
    forecast_pill = (
        "<span class=\"pill\">Prediksi masa depan</span>" if is_future_year else ""
    )

    # Dense, scannable result card
    html = f"""
    <div class="result-card">
      <div class="result-topline">
        <span class="status-dot"></span>
        <span>Hasil prediksi siap</span>
      </div>
      <div class="price-display">
        <div class="eyebrow">Estimasi harga bensin</div>
        <div class="price-value">${pred:.3f}</div>
        <div class="price-unit">USD per liter</div>
      </div>
      <div class="metric-grid">
        <div class="metric-cell">
          <span>Model</span>
          <strong>{model_choice}</strong>
        </div>
        <div class="metric-cell">
          <span>R² test 2026</span>
          <strong>{acc_r2_text}</strong>
        </div>
        <div class="metric-cell">
          <span>WAPE test</span>
          <strong style="color:{wape_colour};">{wape_text}</strong>
        </div>
      </div>
      <div class="detail-pills">
        <span class="pill pill-accent">{country}</span>
        <span class="pill">{region}</span>
        <span class="pill">{int(month):02d}/{int(year)}</span>
        {forecast_pill}
        <span class="pill">R² {r2_text}</span>
        <span class="pill">{income_level}</span>
        <span class="pill">{subsidy_level}</span>
      </div>
    </div>
    {warning_html}
    """

    table = pd.DataFrame(
        {
            "Skenario": [
                "Prediksi Anda",
                f"Minimum historis {country}",
                f"Rata-rata historis {country}",
                f"Maksimum historis {country}",
            ],
            "Harga (USD/L)": [
                round(pred, 3),
                round(stats.get("min", float("nan")), 3),
                round(stats.get("mean", float("nan")), 3),
                round(stats.get("max", float("nan")), 3),
            ],
        }
    )
    return html, table


def build_comparison_df() -> pd.DataFrame:
    """Build the metric comparison table for Tab 2."""
    cols = ["Model", "MAE (USD/L)", "RMSE (USD/L)", "R²",
            "WAPE Test (%)", "WAPE Train (%)", "Gap WAPE (%)",
            "Generalisasi", "CV", "NRMSE (%)", "MAPE (%)"]
    rows = []
    for name, entry in STATE.comparison.get("models", {}).items():
        m = entry.get("metrics", {})
        train = entry.get("train_metrics", {})
        generalization = entry.get("generalization", {})
        gap = generalization.get("gap", {})
        r2 = m.get("R2", float("nan"))
        mape = m.get("MAPE", float("nan"))
        rows.append({
            "Model": name,
            "MAE (USD/L)": round(m.get("MAE", float("nan")), 4),
            "RMSE (USD/L)": round(m.get("RMSE", float("nan")), 4),
            "R²": round(r2, 4),
            "WAPE Test (%)": round(m.get("WAPE", float("nan")), 2),
            "WAPE Train (%)": round(train.get("WAPE", float("nan")), 2),
            "Gap WAPE (%)": round(gap.get("WAPE", float("nan")), 2),
            "Generalisasi": generalization.get("status", "n/a"),
            "CV": entry.get("cv_strategy", "n/a"),
            "NRMSE (%)": round(m.get("NRMSE", float("nan")), 2),
            "MAPE (%)": round(mape, 2),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def _model_card_html(name: str, is_best: bool = False) -> str:
    """Build a single model metric card as HTML."""
    entry = STATE.comparison.get("models", {}).get(name, {})
    m = entry.get("metrics", {})
    r2 = m.get("R2")
    wape = m.get("WAPE")
    mae = m.get("MAE")
    rmse = m.get("RMSE")
    generalization = entry.get("generalization", {})
    gap = generalization.get("gap", {})

    acc_r2 = f"{r2:.4f}" if r2 is not None else "n/a"
    wape_text = f"{wape:.1f}%" if wape is not None else "n/a"
    mae_text = f"{mae:.4f}" if mae is not None else "n/a"
    rmse_text = f"{rmse:.4f}" if rmse is not None else "n/a"
    r2_text = f"{r2:.4f}" if r2 is not None else "n/a"
    gap_text = (
        f"{gap.get('WAPE'):.2f}%" if gap.get("WAPE") is not None else "n/a"
    )
    cv_text = entry.get("cv_strategy", "n/a")

    best_badge = ""
    best_class = ""
    if is_best:
        best_badge = (
            "<span class='model-badge'>Model terbaik</span>"
        )
        best_class = " model-card-best"

    model_descriptions = {
        "KNN": "Memprediksi berdasarkan rata-rata tetangga terdekat dalam ruang fitur",
        "SVM": "Mencari fungsi optimal dengan margin toleransi ε untuk prediksi",
        "Random Forest": "Ansambel ratusan decision tree untuk prediksi yang robust",
    }
    desc = model_descriptions.get(name, "")

    return f"""
    <div class="model-card{best_class}">
      {best_badge}
      <div class="model-title">{name}</div>
      <div class="model-description">{desc}</div>
      <div class="model-primary-metrics">
        <div>
          <span>R² test 2026</span>
          <strong>{acc_r2}</strong>
        </div>
        <div>
          <span>WAPE test</span>
          <strong>{wape_text}</strong>
        </div>
      </div>
      <div class="model-secondary-metrics">
        <div><span>MAE</span><strong>{mae_text}</strong></div>
        <div><span>RMSE</span><strong>{rmse_text}</strong></div>
        <div><span>Gap WAPE</span><strong>{gap_text}</strong></div>
        <div><span>CV</span><strong>{cv_text}</strong></div>
        <div><span>R²</span><strong>{r2_text}</strong></div>
      </div>
    </div>
    """


def best_model_badge() -> str:
    """Return an HTML badge highlighting the best model (lowest RMSE)."""
    best = STATE.comparison.get("best_model")
    if not best:
        return (f"<div class='notice'>"
                f"Belum ada model terlatih. "
                f"Jalankan <code>python train.py</code>.</div>")
    return (
        f"<div class='best-model-callout'>"
        f"<span class='status-dot'></span>"
        f"<strong>{best['name']} direkomendasikan</strong>"
        f"<span>RMSE terendah · {best['RMSE']:.4f}</span>"
        f"</div>"
    )


def descriptive_stats_df() -> pd.DataFrame:
    """Return the descriptive-statistics table for Tab 3."""
    if STATE.df is None:
        return pd.DataFrame()
    return descriptive_stats(STATE.df)


# --------------------------------------------------------------------------- #
# Custom CSS — Mintlify-inspired design system
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
:root {
  --bg: #F0EAEB;
  --surface: #FFFFFF;
  --border: #E0DADA;
  --header-bg: #2D0A12;
  --text-primary: #1A1A2E;
  --text-secondary: #555555;
  --text-muted: #888888;
  --accent: #C41230;
  --accent-light: #FFF0F2;
  --accent-dark: #9E0E27;
  --secondary: #005BAA;
  --secondary-light: #EBF4FF;
  --success: #009A44;
  --success-light: #E8F8EE;
}

/* ---- Global ---- */
body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-primary) !important;
  max-width: 1100px !important;
  margin: 0 auto !important;
}

/* ---- Typography ---- */
h1, h2, h3, h4, .app-title {
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
  font-weight: 700 !important;
}

/* ---- Header ---- */
#app-header {
  background: var(--header-bg);
  border: none;
  border-radius: 12px;
  padding: 28px 32px;
  margin-bottom: 20px;
}
#app-header h1 {
  color: #FFFFFF !important;
  margin: 0 !important;
  font-size: 24px !important;
  font-weight: 700 !important;
  line-height: 1.3;
}
#app-header p {
  color: rgba(255,255,255,0.75) !important;
  margin: 6px 0 0 0 !important;
  font-size: 14px !important;
  line-height: 1.5;
}
#app-header .meta {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  flex-wrap: wrap;
}
#app-header .meta span {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  color: rgba(255,255,255,0.7);
  font-size: 11px;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 6px;
  letter-spacing: 0.02em;
}
#app-header .meta span.accent-tag {
  background: var(--accent);
  border-color: var(--accent);
  color: #FFFFFF;
  font-weight: 600;
}

/* ---- Tabs ---- */
.tab-nav, div[role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  background: transparent !important;
}
.tab-nav button, div[role="tablist"] button {
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 16px !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
  background: transparent !important;
}
.tab-nav button:hover, div[role="tablist"] button:hover {
  color: var(--accent) !important;
}
button.selected, .tab-nav button.selected,
div[role="tablist"] button[aria-selected="true"] {
  color: var(--accent) !important;
  font-weight: 600 !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
}

/* ---- Cards ---- */
.ds-card {
  background: var(--surface) !important;
  border-radius: 12px !important;
  padding: 24px !important;
  border: 1px solid var(--border) !important;
}

/* ---- Section headers inside cards ---- */
.section-label {
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  margin-bottom: 8px !important;
  padding-bottom: 6px !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ---- Primary button ---- */
#predict-btn button, #predict-btn,
button.primary, .gr-button-primary, button[variant="primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: 12px 28px !important;
  font-size: 14px !important;
  transition: background 0.15s ease !important;
  cursor: pointer !important;
}
#predict-btn button:hover, button.primary:hover,
.gr-button-primary:hover {
  background: var(--accent-dark) !important;
}

/* ---- Input elements ---- */
.gradio-container input, .gradio-container select,
.gradio-container textarea {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container input:focus, .gradio-container select:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(196, 18, 48, 0.1) !important;
}

/* ---- Sliders ---- */
.gradio-container input[type="range"]::-webkit-slider-thumb {
  background: var(--accent) !important;
}
.gradio-container input[type="range"]::-moz-range-thumb {
  background: var(--accent) !important;
}
.gradio-container .range-slider {
  accent-color: var(--accent) !important;
}
.gradio-container input[type="range"] {
  accent-color: #C41230 !important;
}

/* ---- Radio / Checkbox ---- */
.gradio-container input[type="radio"]:checked {
  accent-color: #C41230 !important;
  background-color: #C41230 !important;
  border-color: #C41230 !important;
}

/* ---- Dataframes ---- */
.gradio-container table {
  border-radius: 8px !important;
  overflow: hidden !important;
  font-size: 13px !important;
}
.gradio-container th {
  background: var(--bg) !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  font-size: 12px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.03em !important;
}
.gradio-container td {
  color: var(--text-secondary) !important;
}

/* ---- Accordion ---- */
.gradio-container .accordion {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

/* ---- Image containers ---- */
.gradio-container .image-container, .gradio-image {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}

/* ---- Links ---- */
a { color: var(--secondary) !important; }

/* ---- Footer ---- */
#app-footer {
  text-align: center;
  background: var(--header-bg);
  color: rgba(255,255,255,0.65);
  font-size: 12px;
  padding: 20px 28px;
  border-radius: 12px;
  margin-top: 24px;
  border: none;
}

/* ---- Hide unnecessary Gradio chrome ---- */
footer.svelte-1rjryqp { display: none !important; }
.built-with { display: none !important; }

/* ---- Markdown overrides ---- */
.gradio-container .prose h3 {
  font-family: 'Inter', sans-serif !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin-bottom: 4px !important;
}
.gradio-container .prose p, .gradio-container .prose li {
  color: var(--text-secondary) !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
}
.gradio-container .prose blockquote {
  border-left: 3px solid var(--secondary) !important;
  background: var(--secondary-light) !important;
  padding: 12px 16px !important;
  border-radius: 0 8px 8px 0 !important;
  color: var(--text-secondary) !important;
  font-size: 13px !important;
}
"""

# The later layer intentionally overrides Gradio defaults and the legacy rules
# above while keeping compatibility with both Gradio 4 and Gradio 6.
CUSTOM_CSS += """
:root {
  --bg: #f7f7f7;
  --surface: #ffffff;
  --surface-soft: #fafafa;
  --surface-code: #1c1c1e;
  --border: #e5e5e5;
  --border-soft: #ededed;
  --text-primary: #0a0a0a;
  --text-secondary: #5a5a5c;
  --text-muted: #888888;
  --accent: #00d4a4;
  --accent-light: #e8fbf6;
  --accent-dark: #00b48a;
  --secondary: #3772cf;
  --secondary-light: #eef4ff;
  --success: #1ba673;
  --success-light: #e8f8f2;
}
body.dark {
  --body-background-fill: #ffffff;
  --background-fill-primary: #ffffff;
  --background-fill-secondary: #f7f7f7;
  --panel-background-fill: #ffffff;
  --block-background-fill: #ffffff;
  --block-border-color: #e5e5e5;
  --block-label-background-fill: #ffffff;
  --block-label-border-color: #e5e5e5;
  --block-label-text-color: #5a5a5c;
  --block-title-text-color: #0a0a0a;
  --block-info-text-color: #888888;
  --input-background-fill: #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-background-fill-focus: #ffffff;
  --input-border-color: #e5e5e5;
  --input-border-color-hover: #a8a8aa;
  --input-border-color-focus: #00b48a;
  --body-text-color: #0a0a0a;
  --body-text-color-subdued: #888888;
  --checkbox-background-color: #ffffff;
  --checkbox-background-color-hover: #ffffff;
  --checkbox-background-color-focus: #ffffff;
  --checkbox-label-background-fill: #ffffff;
  --checkbox-label-background-fill-selected: #f7f7f7;
  --checkbox-label-text-color: #5a5a5c;
  --checkbox-label-text-color-selected: #0a0a0a;
  --checkbox-label-border-color-selected: #e5e5e5;
  --table-text-color: #5a5a5c;
  --table-even-background-fill: #ffffff;
  --table-odd-background-fill: #fafafa;
  --accordion-text-color: #0a0a0a;
}

html {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
body {
  background: var(--surface) !important;
}
.gradio-container {
  background: var(--surface) !important;
  color: var(--text-primary) !important;
  max-width: 1280px !important;
  padding: 0 28px 40px !important;
}
.gradio-container * {
  box-sizing: border-box;
}
.gradio-container .gap {
  gap: 16px !important;
}
.gradio-container .form {
  background: transparent !important;
  border: 0 !important;
}
h1, h2, h3, h4, .app-title {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  text-wrap: balance;
}
.gradio-container .prose h3 {
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 18px !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
  margin: 0 0 4px !important;
}
.gradio-container .prose p,
.gradio-container .prose li {
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  line-height: 1.55 !important;
  text-wrap: pretty;
}
.gradio-container code,
.gradio-container .prose code {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: #1c1c1e !important;
  font-family: 'Geist Mono', Consolas, monospace !important;
  font-size: 12px !important;
  padding: 2px 6px !important;
}

#top-nav {
  align-items: center;
  background: rgba(255,255,255,.94);
  border-bottom: 1px solid var(--border-soft);
  display: flex;
  justify-content: space-between;
  min-height: 64px;
  padding: 0 4px;
}
#top-nav .brand {
  align-items: center;
  color: var(--text-primary) !important;
  display: flex;
  font-size: 14px;
  font-weight: 600;
  gap: 10px;
}
#top-nav .brand-logo {
  display: block;
  height: 34px;
  max-width: min(42vw, 210px);
  object-fit: contain;
  object-position: left center;
  width: auto;
}
#top-nav .brand-mark {
  align-items: center;
  background: var(--accent) !important;
  border: 1px solid var(--accent-dark);
  border-radius: 8px;
  color: transparent !important;
  display: inline-flex;
  font-family: 'Geist Mono', monospace;
  font-size: 0;
  font-weight: 600;
  height: 30px;
  justify-content: center;
  letter-spacing: .04em;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.28);
  width: 30px;
}
#top-nav .brand-mark::after {
  color: #0a0a0a !important;
  content: "FP";
  font-family: 'Geist Mono', Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
}
#top-nav .nav-meta {
  align-items: center;
  color: var(--text-secondary) !important;
  display: flex;
  font-size: 12px;
  gap: 16px;
}
#top-nav .live-pill {
  align-items: center;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-primary) !important;
  display: inline-flex;
  font-weight: 500;
  gap: 7px;
  padding: 6px 10px;
}
#top-nav .brand > span:not(.brand-mark) {
  color: var(--text-primary) !important;
}
#top-nav .nav-meta > span:not(.live-pill) {
  color: var(--text-secondary) !important;
}
.status-dot {
  background: var(--accent);
  border-radius: 999px;
  display: inline-block;
  height: 7px;
  width: 7px;
}

#app-header {
  background:
    radial-gradient(circle at 78% 12%, rgba(0,212,164,.20), transparent 28%),
    linear-gradient(135deg, #f4f9fa 0%, #f7f3ec 100%);
  border: 1px solid var(--border-soft);
  border-radius: 12px;
  margin: 24px 0 20px;
  overflow: hidden;
  padding: 42px 44px;
}
#app-header h1 {
  color: var(--text-primary) !important;
  font-size: clamp(32px, 5vw, 54px) !important;
  font-weight: 600 !important;
  letter-spacing: -1.8px !important;
  line-height: 1.08 !important;
  margin: 0 !important;
  max-width: 760px;
}
#app-header p {
  color: var(--text-secondary) !important;
  font-size: 16px !important;
  line-height: 1.55;
  margin: 14px 0 0 !important;
  max-width: 660px;
}
#app-header .eyebrow {
  color: var(--text-primary);
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .08em;
  margin-bottom: 16px;
  text-transform: uppercase;
}
#app-header .meta {
  display: flex;
  gap: 8px;
  margin-top: 24px;
}
#app-header .meta span {
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(10,10,10,.10);
  border-radius: 999px;
  color: var(--text-secondary);
  font-size: 12px;
  padding: 7px 12px;
}
#app-header .meta span.accent-tag {
  background: var(--text-primary);
  border-color: var(--text-primary);
  color: var(--surface) !important;
}
#app-header .meta .status-dot {
  background: var(--accent) !important;
  border: 0 !important;
  height: 7px;
  padding: 0 !important;
  width: 7px;
}

.tab-nav, div[role="tablist"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 4px !important;
  margin-bottom: 24px !important;
}
.tab-nav button, div[role="tablist"] button {
  background: transparent !important;
  border: 0 !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  min-height: 44px !important;
  padding: 10px 14px !important;
}
.tab-nav button:hover, div[role="tablist"] button:hover {
  color: var(--text-primary) !important;
}
button.selected, .tab-nav button.selected,
div[role="tablist"] button[aria-selected="true"] {
  background: transparent !important;
  border-bottom-color: var(--text-primary) !important;
  color: var(--text-primary) !important;
}

.ds-card, .content-panel, .results-panel {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 24px !important;
}
.results-panel {
  background: var(--surface-soft) !important;
}
.page-intro {
  margin-bottom: 18px !important;
}
.page-intro h3 {
  font-size: 24px !important;
  letter-spacing: -.4px !important;
}
.page-intro p {
  margin: 4px 0 0 !important;
  max-width: 720px;
}
.form-section {
  align-items: baseline;
  border-bottom: 1px solid var(--border-soft);
  display: flex;
  gap: 10px;
  justify-content: space-between;
  margin: 20px 0 8px;
  padding-bottom: 8px;
}
.form-section span {
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 600;
}
.form-section small {
  color: var(--text-muted);
  font-size: 11px;
  text-align: right;
}

.gradio-container label span {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap {
  border-color: var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container input:focus,
.gradio-container select:focus,
.gradio-container textarea:focus,
.gradio-container .wrap:focus-within {
  border-color: var(--accent-dark) !important;
  box-shadow: 0 0 0 3px rgba(0,212,164,.12) !important;
}
.gradio-container input[type="range"] {
  accent-color: var(--text-primary) !important;
}
.gradio-container input[type="radio"]:checked {
  accent-color: var(--text-primary) !important;
  background-color: var(--text-primary) !important;
  border-color: var(--text-primary) !important;
}
body.dark .gradio-container .block,
body.dark .gradio-container .wrap,
body.dark .gradio-container .form,
body.dark .gradio-container .table-wrap {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
}
body.dark .gradio-container input,
body.dark .gradio-container select,
body.dark .gradio-container textarea {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
}
body.dark .gradio-container label,
body.dark .gradio-container label span,
body.dark .gradio-container .wrap span {
  color: var(--text-secondary) !important;
}
body.dark .gradio-container label:has(input[type="radio"]) {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text-secondary) !important;
}
body.dark .gradio-container label:has(input[type="radio"]:checked) {
  background: var(--bg) !important;
  border-color: var(--text-primary) !important;
  color: var(--text-primary) !important;
}
body.dark .gradio-container table,
body.dark .gradio-container th,
body.dark .gradio-container td {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text-secondary) !important;
}
#predict-btn, #predict-btn button,
button.primary, .gr-button-primary, button[variant="primary"] {
  background: var(--text-primary) !important;
  border: 1px solid var(--text-primary) !important;
  border-radius: 8px !important;
  color: var(--surface) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  min-height: 44px !important;
  transition: transform .15s ease, background-color .15s ease, box-shadow .15s ease !important;
}
#predict-btn:hover, #predict-btn button:hover,
button.primary:hover, .gr-button-primary:hover {
  background: #1c1c1e !important;
  box-shadow: 0 8px 20px rgba(0,0,0,.12) !important;
}
#predict-btn:active, #predict-btn button:active, button.primary:active {
  transform: scale(.98);
}

.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.result-topline {
  align-items: center;
  border-bottom: 1px solid var(--border-soft);
  color: var(--text-secondary);
  display: flex;
  font-size: 12px;
  gap: 8px;
  padding: 12px 16px;
}
.price-display {
  padding: 32px 24px 28px;
  text-align: center;
}
.eyebrow {
  color: var(--text-muted);
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: .07em;
  text-transform: uppercase;
}
.price-value {
  color: var(--text-primary);
  font-family: 'Geist Mono', monospace;
  font-size: clamp(42px, 7vw, 62px);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  letter-spacing: -2px;
  line-height: 1.05;
  margin-top: 10px;
}
.price-unit {
  color: var(--text-muted);
  font-size: 13px;
  margin-top: 6px;
}
.metric-grid {
  border-bottom: 1px solid var(--border-soft);
  border-top: 1px solid var(--border-soft);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
}
.metric-cell {
  padding: 16px;
  text-align: center;
}
.metric-cell + .metric-cell {
  border-left: 1px solid var(--border-soft);
}
.metric-cell span,
.model-primary-metrics span,
.model-secondary-metrics span {
  color: var(--text-muted);
  display: block;
  font-size: 10px;
  letter-spacing: .05em;
  margin-bottom: 4px;
  text-transform: uppercase;
}
.metric-cell strong,
.model-primary-metrics strong,
.model-secondary-metrics strong {
  color: var(--text-primary);
  font-family: 'Geist Mono', monospace;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.detail-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  padding: 16px;
}
.pill {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 11px;
  padding: 5px 9px;
}
.pill-accent {
  background: var(--accent-light);
  border-color: rgba(0,180,138,.28);
  color: #08785e;
  font-weight: 600;
}
.pill-error {
  background: #fff3f3;
  border-color: #f2caca;
  color: #d45656;
}
.market-table {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
.market-table .table-wrap,
.market-table .virtual-table-viewport {
  background: var(--surface) !important;
  border-radius: 6px !important;
  overflow: hidden !important;
}
.market-table table,
.market-table .header-table {
  border: 0 !important;
  border-radius: 0 !important;
}
.market-table thead,
.market-table thead tr,
.market-table thead th,
.market-table .header-cell {
  background: var(--bg) !important;
  color: var(--text-primary) !important;
}
.market-table .header-cell,
.market-table .header-cell *,
.market-table thead th,
.market-table thead th * {
  border-radius: 2px !important;
}
.market-table .virtual-row,
.market-table .virtual-row .body-cell {
  background: var(--surface) !important;
  color: var(--text-secondary) !important;
}
.market-table .virtual-row:nth-of-type(even),
.market-table .virtual-row:nth-of-type(even) .body-cell {
  background: var(--surface-soft) !important;
}
.market-table .body-cell {
  border-bottom: 1px solid var(--border-soft) !important;
  border-left: 0 !important;
  border-right: 1px solid var(--border-soft) !important;
  border-top: 0 !important;
  padding: 10px 12px !important;
}
.market-table .body-cell:last-child {
  border-right: 0 !important;
}
.market-table .body-cell .cell-wrap,
.market-table .body-cell [role="button"],
.market-table .body-cell span {
  background: transparent !important;
  border: 0 !important;
  border-radius: 2px !important;
  box-shadow: none !important;
  color: var(--text-secondary) !important;
  min-height: 0 !important;
  padding: 0 !important;
}
.market-table .body-cell.first-column [role="button"],
.market-table .body-cell.first-column span {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}
.empty-state {
  align-items: center;
  color: var(--text-muted);
  display: flex;
  flex-direction: column;
  font-size: 13px;
  justify-content: center;
  min-height: 300px;
  padding: 48px 24px;
  text-align: center;
}
.empty-state span {
  color: var(--text-muted) !important;
}
.empty-state-mark {
  align-items: center;
  background: var(--text-primary);
  border-radius: 10px;
  color: var(--surface);
  display: flex;
  font-family: 'Geist Mono', monospace;
  font-size: 13px;
  height: 40px;
  justify-content: center;
  margin-bottom: 14px;
  width: 40px;
}
.empty-state strong {
  color: var(--text-primary);
}

.best-model-callout {
  align-items: center;
  background: var(--accent-light);
  border: 1px solid rgba(0,180,138,.25);
  border-radius: 8px;
  color: var(--text-secondary);
  display: flex;
  flex-wrap: wrap;
  font-size: 13px;
  gap: 9px;
  margin-bottom: 16px;
  padding: 12px 14px;
}
.best-model-callout strong {
  color: var(--text-primary);
}
.model-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  min-height: 100%;
  padding: 22px;
  position: relative;
}
.model-card-best {
  border: 2px solid var(--accent);
  box-shadow: 0 8px 24px rgba(0,212,164,.08);
}
.model-badge {
  background: var(--accent);
  border-radius: 999px;
  color: var(--text-primary);
  font-size: 10px;
  font-weight: 600;
  padding: 4px 8px;
  position: absolute;
  right: 14px;
  top: 14px;
}
.model-title {
  color: var(--text-primary);
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 5px;
}
.model-description {
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
  min-height: 54px;
  padding-right: 34px;
}
.model-primary-metrics {
  display: grid;
  gap: 8px;
  grid-template-columns: 1fr 1fr;
  margin-top: 16px;
}
.model-primary-metrics > div {
  background: var(--bg);
  border-radius: 8px;
  padding: 12px;
}
.model-primary-metrics strong {
  font-size: 17px;
}
.model-secondary-metrics {
  border-top: 1px solid var(--border-soft);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  margin-top: 14px;
  padding-top: 14px;
  text-align: center;
}
.notice {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-secondary);
  font-size: 13px;
  padding: 14px 16px;
}
.notice p {
  margin: 6px 0 0;
}
.notice-error {
  background: #fff7f7;
  border-color: #f0cccc;
}
.notice-title {
  color: #d45656;
  font-weight: 600;
}
.snapshot-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(4, 1fr);
  margin-bottom: 22px;
}
.snapshot-item {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.snapshot-item span {
  color: var(--text-muted);
  display: block;
  font-size: 10px;
  letter-spacing: .05em;
  margin-bottom: 5px;
  text-transform: uppercase;
}
.snapshot-item strong {
  color: var(--text-primary);
  font-family: 'Geist Mono', monospace;
  font-size: 17px;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.gradio-container table {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
}
.gradio-container th {
  background: var(--bg) !important;
  color: var(--text-primary) !important;
  font-size: 11px !important;
}
.gradio-container td {
  color: var(--text-secondary) !important;
  font-variant-numeric: tabular-nums;
}
.dataset-table {
  background: var(--surface) !important;
  border: 1px solid #c9eee4 !important;
  border-radius: 6px !important;
  box-shadow: 0 8px 24px rgba(0, 180, 138, .06);
  overflow: hidden !important;
}
.dataset-table .table-wrap {
  background: var(--surface) !important;
  border-radius: 5px !important;
}
.dataset-table table {
  background: var(--surface) !important;
  border: 0 !important;
  border-collapse: separate !important;
  border-radius: 4px !important;
  border-spacing: 0 !important;
  overflow: hidden !important;
}
.dataset-table thead,
.dataset-table thead tr,
.dataset-table thead th {
  background: #dff8f1 !important;
}
.dataset-table thead th {
  border-bottom: 1px solid #a8dfd1 !important;
  border-right: 1px solid #c7eadf !important;
  color: #075e4d !important;
  font-size: 11px !important;
  letter-spacing: .045em;
  padding: 13px 12px !important;
}
.dataset-table .header-cell,
.dataset-table .header-cell .cell-wrap,
.dataset-table .header-cell .header-content,
.dataset-table .header-cell span {
  background: transparent !important;
  border: 0 !important;
  border-radius: 1px !important;
  box-shadow: none !important;
  color: #075e4d !important;
}
.dataset-table tbody tr:nth-child(odd),
.dataset-table tbody tr:nth-child(odd) td {
  background: #f5fcfa !important;
}
.dataset-table tbody tr:nth-child(even),
.dataset-table tbody tr:nth-child(even) td {
  background: #fffaf1 !important;
}
.dataset-table tbody td {
  border-bottom: 1px solid #e3f1ed !important;
  border-right: 1px solid #edf3f1 !important;
  color: #3a3a3c !important;
  padding: 10px 12px !important;
}
.dataset-table tbody tr:last-child td {
  border-bottom: 0 !important;
}
.dataset-table th:last-child,
.dataset-table td:last-child {
  border-right: 0 !important;
}
.dataset-table td button,
.dataset-table th button,
.dataset-table td input,
.dataset-table td div {
  background: transparent !important;
  border: 0 !important;
  border-radius: 2px !important;
  box-shadow: none !important;
  color: inherit !important;
}
.dataset-table tbody td:first-child,
.dataset-table tbody td:first-child button {
  color: #08785e !important;
  font-weight: 600 !important;
}
.dataset-table tbody td:not(:first-child),
.dataset-table tbody td:not(:first-child) button {
  font-family: 'Geist Mono', Consolas, monospace !important;
  font-variant-numeric: tabular-nums;
}
.dataset-table .virtual-body,
.dataset-table .virtual-table-viewport {
  background: var(--surface) !important;
}
.dataset-table .virtual-table-viewport {
  border: 1px solid #a8dfd1 !important;
  border-radius: 4px !important;
  overflow: hidden !important;
}
.dataset-table .header-table {
  border: 0 !important;
  border-radius: 0 !important;
}
.dataset-table .virtual-row:nth-of-type(odd),
.dataset-table .virtual-row:nth-of-type(odd) .body-cell {
  background: #f5fcfa !important;
}
.dataset-table .virtual-row:nth-of-type(even),
.dataset-table .virtual-row:nth-of-type(even) .body-cell {
  background: #fffaf1 !important;
}
.dataset-table .virtual-row {
  color: #3a3a3c !important;
}
.dataset-table .body-cell {
  border-left: 0 !important;
  border-top: 0 !important;
  border-bottom: 1px solid #e3f1ed !important;
  border-right: 1px solid #edf3f1 !important;
  color: #3a3a3c !important;
  padding: 10px 12px !important;
}
.dataset-table .body-cell:last-child {
  border-right: 0 !important;
}
.dataset-table .body-cell .cell-wrap,
.dataset-table .body-cell [role="button"],
.dataset-table .body-cell span {
  background: transparent !important;
  border: 0 !important;
  border-radius: 1px !important;
  box-shadow: none !important;
  color: #3a3a3c !important;
  min-height: 0 !important;
  padding: 0 !important;
}
.dataset-table .body-cell.first-column .cell-wrap,
.dataset-table .body-cell.first-column [role="button"],
.dataset-table .body-cell.first-column span {
  color: #08785e !important;
  font-weight: 600 !important;
}
.dataset-table .body-cell:not(.first-column) .cell-wrap,
.dataset-table .body-cell:not(.first-column) [role="button"],
.dataset-table .body-cell:not(.first-column) span {
  font-family: 'Geist Mono', Consolas, monospace !important;
  font-variant-numeric: tabular-nums;
}
.gradio-container .accordion {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.model-explanation {
  background: var(--surface) !important;
  border-color: var(--border) !important;
}
.model-explanation,
.model-explanation summary,
.model-explanation button,
.model-explanation .label-wrap,
.model-explanation .prose,
.model-explanation .prose p,
.model-explanation .prose li,
.model-explanation .prose span {
  color: var(--text-secondary) !important;
}
.model-explanation summary,
.model-explanation button,
.model-explanation .label-wrap {
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}
.model-explanation .prose strong {
  color: var(--text-primary) !important;
  font-style: normal !important;
  font-weight: 600 !important;
}
.model-explanation .prose em {
  color: var(--text-secondary) !important;
  font-style: italic !important;
}
.model-explanation svg {
  color: var(--text-secondary) !important;
  fill: currentColor !important;
}
.gradio-container .image-container, .gradio-image {
  background: var(--surface-soft) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.dataset-chart {
  background: var(--surface) !important;
  border-color: #c9eee4 !important;
}
.dataset-chart .block-label,
.dataset-chart .block-label span,
.dataset-chart .block-label p,
.dataset-chart .label-wrap,
.dataset-chart .label-wrap span,
.dataset-chart label.float,
.dataset-chart label[data-testid="block-label"],
.dataset-chart label[data-testid="block-label"] span {
  background: #1c1c1e !important;
  color: #ffffff !important;
}
.dataset-chart .block-label {
  border: 1px solid #2f2f31 !important;
  border-radius: 4px 4px 0 0 !important;
  font-weight: 500 !important;
}
.dataset-chart .block-label svg,
.dataset-chart .label-wrap svg,
.dataset-chart label[data-testid="block-label"] svg {
  color: var(--accent) !important;
  fill: currentColor !important;
  stroke: currentColor !important;
}
.dataset-chart label[data-testid="block-label"] {
  border: 1px solid #2f2f31 !important;
  border-radius: 4px 4px 0 0 !important;
  font-weight: 500 !important;
}
.dataset-chart .icon-button,
.dataset-chart .icon-button div {
  background: #1c1c1e !important;
  color: #ffffff !important;
}
.dataset-chart .icon-button svg {
  color: #ffffff !important;
  fill: none !important;
  stroke: currentColor !important;
}
.dataset-chart .icon-button:hover,
.dataset-chart .icon-button:hover div {
  background: #303033 !important;
}
.gradio-container img {
  outline: 1px solid rgba(0,0,0,.06);
  outline-offset: -1px;
}
.gradio-container .prose blockquote {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--accent-dark) !important;
  border-radius: 0 8px 8px 0 !important;
  color: var(--text-secondary) !important;
  margin: 12px 0 18px !important;
  padding: 12px 14px !important;
}
.gradio-container .prose blockquote,
.gradio-container .prose blockquote p,
.gradio-container .prose blockquote strong,
.gradio-container .prose blockquote em {
  color: var(--text-secondary) !important;
}
.gradio-container .prose blockquote strong {
  color: var(--text-primary) !important;
}

#app-footer {
  align-items: center;
  background: transparent;
  border-top: 1px solid var(--border);
  border-radius: 0;
  color: var(--text-muted);
  display: flex;
  font-size: 12px;
  justify-content: space-between;
  margin-top: 32px;
  padding: 22px 2px 0;
  text-align: left;
}
#app-footer strong {
  color: var(--text-primary);
  font-weight: 600;
}
#app-footer span {
  color: var(--text-muted) !important;
}
footer {
  display: none !important;
}

/* ---- Fullscreen application shell ---- */
html, body {
  margin: 0 !important;
  max-width: none !important;
  min-height: 0;
  width: 100%;
}
#root, gradio-app {
  display: block;
  max-width: none !important;
  min-height: 0;
  width: 100% !important;
}
.gradio-container {
  box-sizing: border-box;
  margin: 0 !important;
  max-width: none !important;
  min-height: 0 !important;
  padding: 0 clamp(16px, 2.2vw, 44px) 32px !important;
  width: 100% !important;
}
#top-nav {
  margin-inline: calc(clamp(16px, 2.2vw, 44px) * -1);
  padding-inline: clamp(16px, 2.2vw, 44px);
  position: sticky;
  top: 0;
  z-index: 20;
}
#app-header {
  border-radius: 0 0 12px 12px;
  margin-inline: calc(clamp(16px, 2.2vw, 44px) * -1);
  margin-top: 0;
  padding: clamp(36px, 5vw, 72px) clamp(24px, 5vw, 88px);
}
#app-header h1 {
  max-width: 920px;
}
#app-header p {
  max-width: 760px;
}
.prediction-workspace {
  align-items: stretch !important;
}
.prediction-workspace > .ds-card,
.prediction-workspace > .results-panel {
  height: fit-content;
}
.prediction-workspace > .results-panel {
  align-self: flex-start !important;
}
.prediction-workspace > .results-panel .empty-state {
  min-height: min(42vh, 420px);
}

@media (max-width: 768px) {
  .gradio-container {
    padding: 0 14px 28px !important;
  }
  #top-nav {
    min-height: 56px;
  }
  #top-nav .nav-meta > span:not(.live-pill) {
    display: none;
  }
  #app-header {
    border-radius: 0 0 12px 12px;
    margin-inline: -14px;
    margin-top: 14px;
    padding: 28px 22px;
  }
  #top-nav {
    margin-inline: -14px;
    padding-inline: 14px;
  }
  .prediction-workspace {
    min-height: auto;
  }
  .prediction-workspace > .results-panel {
    min-height: auto;
    position: static;
  }
  #app-header h1 {
    font-size: 34px !important;
    letter-spacing: -1px !important;
  }
  #app-header p {
    font-size: 14px !important;
  }
  .ds-card, .content-panel, .results-panel {
    padding: 18px !important;
  }
  .metric-grid {
    grid-template-columns: 1fr;
  }
  .metric-cell + .metric-cell {
    border-left: 0;
    border-top: 1px solid var(--border-soft);
  }
  .form-section {
    align-items: flex-start;
    flex-direction: column;
    gap: 2px;
  }
  .form-section small {
    text-align: left;
  }
  .snapshot-grid {
    grid-template-columns: 1fr 1fr;
  }
  #app-footer {
    align-items: flex-start;
    flex-direction: column;
    gap: 6px;
  }
}
"""

# --------------------------------------------------------------------------- #
# Gradio theme — Mint accent, neutral product surfaces
# --------------------------------------------------------------------------- #
MINT_THEME = gr.themes.Default(
    primary_hue=gr.themes.Color(
        c50="#E8FBF6", c100="#CCF7ED", c200="#9EEEDC",
        c300="#65E3C7", c400="#2AD9B4", c500="#00D4A4",
        c600="#00B48A", c700="#07876B", c800="#096B56",
        c900="#085747", c950="#033129",
    ),
    neutral_hue=gr.themes.Color(
        c50="#FAFAFA", c100="#F7F7F7", c200="#EDEDED",
        c300="#E5E5E5", c400="#A8A8AA", c500="#888888",
        c600="#5A5A5C", c700="#3A3A3C", c800="#1C1C1E",
        c900="#111111", c950="#0A0A0A",
    ),
)

# --------------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks app.

    On Gradio < 6 the design-system ``css``/``theme`` are passed to the Blocks
    constructor (required by the HF 4.x runtime); on Gradio >= 6 they are
    deferred to :meth:`launch` (see :func:`launch_app`).
    """
    blocks_kwargs = {"title": "Global Fuel Price Predictor"}
    default_country = (
        STATE.countries[0] if STATE.countries else "United States"
    )
    default_region, default_income, default_subsidy = country_profile_values(
        default_country)
    default_brent_range = STATE.brent_year_ranges.get(
        STATE.year_max, (100.0, 130.0))
    default_brent = round(sum(default_brent_range) / 2, 1)
    default_model = STATE.comparison.get("best_model", {}).get(
        "name", "Random Forest")
    if GRADIO_MAJOR < 6:
        blocks_kwargs["css"] = CUSTOM_CSS
        blocks_kwargs["theme"] = MINT_THEME
    with gr.Blocks(**blocks_kwargs) as demo:

        # ---- Product navigation and hero ---- #
        gr.HTML(
            "<link rel='preconnect' href='https://fonts.googleapis.com'>"
            "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
            "family=Geist+Mono:wght@400;500;600&"
            "family=Inter:wght@400;500;600;700&display=swap'>"
            "<div id='top-nav'>"
            f"<div class='brand'><img class='brand-logo' src='{BRAND_LOGO_DATA_URI}' "
            "alt='FuelPredict'></div>"
            "<div class='nav-meta'><span>Dasar Ilmu Data</span>"
            "<span>2020–2026</span>"
            f"<span>Prediksi sampai {STATE.forecast_year_max}</span>"
            "<span class='live-pill'><span class='status-dot'></span>"
            "Model siap</span></div></div>"
            "<div id='app-header'>"
            "<div class='eyebrow'>Machine learning dashboard</div>"
            "<h1>Estimasi harga bensin global, tanpa tebakan rumit.</h1>"
            "<p>Pilih negara dan kondisi pasar, lalu bandingkan hasil prediksi "
            "dengan rentang harga historis negara tersebut.</p>"
            "<div class='meta'>"
            "<span class='accent-tag'><span class='status-dot'></span> Siap digunakan</span>"
            "<span>84 negara</span><span>3 model regresi</span>"
            "<span>Harga dalam USD/liter</span>"
            "</div>"
            "</div>"
        )

        if not STATE.ready:
            gr.HTML(_error_card(STATE.load_error or "Artifacts gagal dimuat."))

        with gr.Tabs():
            # ---------------------------------------------------------- #
            # TAB 1 — Prediction
            # ---------------------------------------------------------- #
            with gr.Tab("Prediksi Harga"):
                gr.Markdown(
                    "### Buat estimasi baru\n"
                    "Atur parameter skenario di kiri. Ringkasan hasil dan konteks "
                    "pasar akan muncul di kanan.",
                    elem_classes="page-intro",
                )
                with gr.Row(elem_classes="prediction-workspace"):
                    # ---- Left: Input panel ---- #
                    with gr.Column(scale=5, min_width=400, elem_classes="ds-card"):
                        gr.Markdown("### Parameter skenario")
                        gr.Markdown(
                            "Gunakan nilai yang paling mendekati kondisi yang "
                            "ingin dianalisis.")

                        # Section: Lokasi
                        gr.HTML(_section_label(
                            "01 · Lokasi", "Menentukan pola harga dasar"))
                        with gr.Row():
                            country = gr.Dropdown(
                                choices=STATE.countries or ["United States"],
                                value=default_country,
                                label="Negara", filterable=True)
                            region = gr.Textbox(
                                value=default_region, label="Wilayah",
                                interactive=False)

                        # Section: Ekonomi
                        gr.HTML(_section_label(
                            "02 · Profil negara", "Otomatis dari dataset"))
                        with gr.Row():
                            income_level = gr.Textbox(
                                value=default_income, label="Tingkat Pendapatan",
                                interactive=False)
                            subsidy_level = gr.Textbox(
                                value=default_subsidy, label="Tingkat Subsidi",
                                interactive=False)

                        # Section: Pasar & Pajak
                        gr.HTML(_section_label(
                            "03 · Pasar & pajak", "Variabel yang paling mudah berubah"))
                        brent_crude = gr.Slider(
                            20, 150, value=default_brent, step=0.5,
                            label="Harga Brent Crude (USD/barel)")
                        tax_pct = gr.Slider(
                            0, 100, value=30, step=0.1,
                            label="Persentase Pajak (%)")

                        # Section: Waktu
                        gr.HTML(_section_label(
                            "04 · Periode", "Waktu estimasi"))
                        with gr.Row():
                            year = gr.Slider(
                                STATE.year_min, STATE.forecast_year_max,
                                value=STATE.year_max, step=1,
                                label=f"Tahun (data sampai {STATE.year_max})")
                            month = gr.Dropdown(
                                choices=MONTHS, value=6, label="Bulan")

                        # Section: Model & Action
                        gr.HTML(_section_label(
                            "05 · Model",
                            f"{default_model} untuk holdout; "
                            f"{FUTURE_RECOMMENDED_MODEL} untuk forecast"))
                        model_choice = gr.Radio(
                            MODEL_DISPLAY, value=default_model,
                            label="Pilih Model")
                        btn_predict = gr.Button(
                            "Hitung prediksi", variant="primary",
                            elem_id="predict-btn")

                    # ---- Right: Results panel ---- #
                    with gr.Column(scale=4, min_width=360, elem_classes="results-panel"):
                        gr.Markdown("### Ringkasan hasil")
                        out_html = gr.HTML(
                            "<div class='empty-state'>"
                            "<div class='empty-state-mark'>USD</div>"
                            "<strong>Belum ada estimasi</strong>"
                            "<span>Lengkapi skenario lalu tekan Hitung prediksi.</span>"
                            "</div>")
                        gr.Markdown("### Posisi terhadap histori negara")
                        out_table = gr.Dataframe(
                            headers=["Skenario", "Harga (USD/L)"],
                            label="",
                            interactive=False, wrap=True,
                            elem_classes="market-table")

                country.change(
                    fn=country_profile_values,
                    inputs=[country],
                    outputs=[region, income_level, subsidy_level],
                )
                btn_predict.click(
                    fn=predict_price,
                    inputs=[country, region, income_level, subsidy_level,
                            brent_crude, tax_pct, year, month, model_choice],
                    outputs=[out_html, out_table],
                )

            # ---------------------------------------------------------- #
            # TAB 2 — Model comparison
            # ---------------------------------------------------------- #
            with gr.Tab("Perbandingan Model"):
                gr.Markdown(
                    "### Performa model\n"
                    "Bandingkan akurasi, ketepatan, dan error sebelum memilih "
                    "model untuk skenario Anda.",
                    elem_classes="page-intro",
                )
                gr.Markdown(
                    "> **Cara baca:** *MAE/RMSE* = error dalam USD/liter "
                    "(makin kecil = makin baik). "
                    "*R²* = variasi yang dijelaskan pada holdout tahun 2026, "
                    "bukan confidence untuk satu prediksi. "
                    "*WAPE/NRMSE* = error relatif terhadap skala harga. "
                    "*Gap WAPE* = selisih error test dan train; makin kecil "
                    "biasanya makin stabil. Semua tuning memakai "
                    "*TimeSeriesSplit* agar urutan waktu tetap terjaga.")
                gr.HTML(best_model_badge())

                # Model cards
                best_name = STATE.comparison.get("best_model", {}).get("name", "")
                with gr.Row():
                    for model_name in MODEL_DISPLAY:
                        with gr.Column():
                            gr.HTML(_model_card_html(
                                model_name,
                                is_best=(model_name == best_name)))

                # Detailed metrics table
                with gr.Accordion("Lihat tabel metrik lengkap", open=False):
                    gr.Dataframe(
                        value=build_comparison_df(),
                        label="",
                        interactive=False, wrap=True)

                # Comparison chart
                gr.Markdown("### Visualisasi Perbandingan")
                gr.Image(value=_img("model_comparison.png"),
                         label="", show_label=False)

                # Model explanations
                gr.Markdown("### Penjelasan Model")
                with gr.Accordion(
                        "KNN — K-Nearest Neighbours", open=False,
                        elem_classes="model-explanation"):
                    gr.Markdown(
                        "**KNN** memprediksi harga dengan merata-ratakan target "
                        "dari *k* tetangga terdekat dalam ruang fitur. "
                        "Pada versi ini KNN memakai bobot uniform agar tidak "
                        "terlalu menghafal titik training.\n\n"
                        "**Kelebihan:** sederhana, non-parametrik, menangkap pola lokal.\n\n"
                        "**Kekurangan:** sensitif terhadap skala fitur & "
                        "*curse of dimensionality*, inferensi lambat untuk data besar.\n\n"
                        "**Dipilih ketika** hubungan fitur–target bersifat "
                        "lokal dan dataset tidak terlalu besar.")
                with gr.Accordion(
                        "SVM — Support Vector Regression", open=False,
                        elem_classes="model-explanation"):
                    gr.Markdown(
                        "**SVR** mencari fungsi yang menyimpang dari target tidak "
                        "lebih dari ε untuk sebanyak mungkin titik, sambil tetap "
                        "se-datar mungkin. Kernel (RBF/linear/poly) memungkinkan "
                        "pemodelan hubungan non-linier.\n\n"
                        "**Kelebihan:** kuat di ruang berdimensi tinggi, "
                        "tahan outlier (margin ε).\n\n"
                        "**Kekurangan:** mahal secara komputasi pada data besar — "
                        "hyperparameter di-*tune* pada 1.500 baris training "
                        "terbaru dengan TimeSeriesSplit, lalu model terbaik "
                        "di-refit pada 10.000 baris training terbaru.\n\n"
                        "**Dipilih ketika** hubungan non-linier dan "
                        "jumlah fitur relatif tinggi.")
                with gr.Accordion(
                        "Random Forest — Ensemble Regression", open=False,
                        elem_classes="model-explanation"):
                    gr.Markdown(
                        "**Random Forest** adalah ansambel banyak *decision tree*. "
                        "Pada project ini model belajar koreksi residual di atas "
                        "tren harga negara, sehingga tetap cocok untuk skenario "
                        "berbasis waktu.\n\n"
                        "**Kelebihan:** akurasi tinggi, menangani interaksi & "
                        "non-linieritas, robust terhadap skala fitur, "
                        "memberi *feature importance*.\n\n"
                        "**Kekurangan:** model besar, kurang interpretable "
                        "dibanding satu pohon.\n\n"
                        "**Dipilih ketika** ingin akurasi tinggi & insight "
                        "pentingnya fitur — biasanya baseline terkuat "
                        "untuk data tabular.")

            # ---------------------------------------------------------- #
            # TAB 3 — Dataset overview
            # ---------------------------------------------------------- #
            with gr.Tab("Dataset"):
                gr.Markdown(
                    "### Dataset overview\n"
                    "Pahami cakupan data dan distribusinya sebelum menafsirkan "
                    "hasil prediksi.",
                    elem_classes="page-intro",
                )
                gr.HTML(
                    "<div class='snapshot-grid'>"
                    f"<div class='snapshot-item'><span>Total baris</span>"
                    f"<strong>{len(STATE.df) if STATE.df is not None else 0:,}</strong></div>"
                    f"<div class='snapshot-item'><span>Negara</span>"
                    f"<strong>{len(STATE.countries)}</strong></div>"
                    f"<div class='snapshot-item'><span>Wilayah</span>"
                    f"<strong>{len(STATE.regions)}</strong></div>"
                    "<div class='snapshot-item'><span>Periode</span>"
                    "<strong>2020–2026</strong></div></div>"
                )
                gr.Markdown(
                    "**Catatan pemakaian data:** model dilatih pada data "
                    "2020-2025 dan diuji pada holdout 2026. Fitur model memakai "
                    "`country`, Brent, pajak, waktu, serta prior/tren negara "
                    "yang dihitung dari data training saja. `diesel_usd_liter` "
                    "dan `lpg_usd_liter` sengaja tidak dipakai sebagai fitur "
                    "karena terlalu dekat dengan target `petrol_usd_liter`.")
                gr.Markdown("### Statistik deskriptif")
                gr.Dataframe(value=descriptive_stats_df(),
                             interactive=False, wrap=True, label="",
                             elem_classes="dataset-table")

                gr.Markdown("### Visualisasi data")
                with gr.Row():
                    with gr.Column():
                        gr.Image(value=_img("petrol_distribution.png"),
                                 label="Distribusi Harga BBM", show_label=True,
                                 elem_classes="dataset-chart")
                    with gr.Column():
                        gr.Image(value=_img("region_distribution.png"),
                                 label="Distribusi per Wilayah", show_label=True,
                                 elem_classes="dataset-chart")
                gr.Image(value=_img("price_timeseries.png"),
                         label="Tren Harga Rata-rata per Bulan (2020–2026)",
                         show_label=True, elem_classes="dataset-chart")

        # ---- Footer ---- #
        gr.HTML(
            "<div id='app-footer'>"
            "<span><strong>Fuel Price Intelligence</strong> · Tugas Besar Dasar Ilmu Data</span>"
            "<span>scikit-learn · Gradio · Global Fuel Prices 2020–2026</span>"
            f"</div>")

    return demo


demo = build_ui()


def launch_app() -> None:
    """Launch the Gradio server on the default port (7860).

    Passes ``css``/``theme`` here on Gradio >= 6 (where the Blocks constructor
    no longer accepts them) and binds ``0.0.0.0:7860`` for Hugging Face Spaces.
    """
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": 7860}
    if GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CUSTOM_CSS
        launch_kwargs["theme"] = MINT_THEME
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
