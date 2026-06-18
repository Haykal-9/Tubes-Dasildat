---
title: Global Fuel Price Predictor
emoji: ⛽
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# ⛽ Global Fuel Price Predictor

Prediksi **harga bensin (petrol, USD/liter)** untuk 84 negara berdasarkan
negara, harga minyak Brent, pajak, dan waktu. Tiga model regresi — **KNN**, **SVM**, dan
**Random Forest** — dilatih, dibandingkan, dan disajikan lewat web app Gradio
yang siap di-deploy ke **Hugging Face Spaces**.

---

## 🚀 Cara pakai web app

1. Buka tab **Prediksi Harga BBM**.
2. Pilih **country**; region, income level, dan subsidy tampil otomatis sebagai
   metadata read-only negara tersebut.
3. Atur slider **Brent Crude (USD/barrel)**, **Tax (%)**, **Year**, dan pilih **Month**.
4. Pilih model (**KNN / SVM / Random Forest**) lalu tekan **Prediksi**.
5. Lihat harga prediksi, badge model, R² temporal-test, serta
   tabel perbandingan terhadap minimum, rata-rata, dan maksimum historis negara.

Tab **Perbandingan Model** menampilkan metrik ketiga model + grafik, dan tab
**Dataset Overview** menampilkan statistik deskriptif serta visualisasi EDA.

---

## 📊 Dataset

| | |
|---|---|
| **File** | `data/global_fuel_prices_2020_2026.csv` |
| **Periode** | Januari 2020 – April 2026 (mingguan) |
| **Coverage** | 84 negara · 7 region · ~27.500 baris |
| **Target** | `petrol_usd_liter` (harga bensin, USD/liter) |

**Fitur yang digunakan:** `country` (One-Hot), `country_price_prior`,
`brent_crude_usd`, `tax_percentage`, `year`, dan `month`.
`country_price_prior` adalah rata-rata historis negara yang dihitung hanya dari
data training 2020–2025; fitur ini menjaga level harga negara tanpa membocorkan
target test 2026. `diesel_usd_liter` dan `lpg_usd_liter` tidak digunakan karena
keduanya adalah target paralel yang hampir identik dengan harga petrol. Region,
income level, dan subsidy hanya metadata negara.

KNN dan SVM menyimpan `StandardScaler` di dalam sklearn `Pipeline`; Random
Forest menggunakan fitur mentah karena tidak berbasis jarak.

---

## 🤖 Tiga model & kapan dipilih

| Model | Tuning | Inti metode | Kapan dipilih |
|---|---|---|---|
| **KNN** | `GridSearchCV` (n_neighbors, weights, metric) | Rata-rata target dari *k* tetangga terdekat | Pola lokal, dataset tidak terlalu besar |
| **SVM (SVR)** | `GridSearchCV` (kernel, C, epsilon, gamma); tune di subset kecil → refit di subsample 10.000; kernel linier via `LinearSVR` | Margin ε + kernel untuk non-linieritas | Hubungan non-linier, dimensi fitur tinggi |
| **Random Forest** | `RandomizedSearchCV` (n_iter=12) | Ansambel decision tree (bagging) | Akurasi tinggi + *feature importance*, baseline kuat data tabular |

---

## 📈 Hasil perbandingan model

> Hasil di bawah berasal dari **year holdout**: model dilatih pada seluruh
> data 2020–2025 dan diuji hanya pada 1.176 baris tahun 2026. Nilai lengkap
> tersimpan di `data/model_comparison.json`.

| Model | MAE (USD/L) | RMSE (USD/L) | R² | WAPE (%) | NRMSE (%) |
|---|---|---|---|---|---|
| KNN | 0.0983 | 0.1512 | 0.9921 | 3.71 | 5.71 |
| **SVM** 🏆 | **0.0327** | **0.0409** | **0.9994** | 1.23 | **1.55** |
| Random Forest | **0.0266** | 0.1102 | 0.9958 | **1.00** | 4.16 |

> **Cara baca:** *MAE/RMSE* adalah error absolut dalam USD/liter. *WAPE/NRMSE*
> adalah error relatif terhadap skala harga. R² bukan confidence untuk satu
> prediksi dan tetap tinggi karena identitas negara menjelaskan sebagian besar
> variasi harga.

> Sebagai pembanding, baseline yang hanya memakai rata-rata harga tiap negara
> sudah mencapai R² **0.9237** pada test 2026. Korelasi petrol dengan diesel dan
> LPG masing-masing **0.9990** dan **0.9999**, sehingga keduanya sengaja dilarang
> menjadi fitur.

> Region, tingkat pendapatan, dan subsidi tidak pernah berubah di dalam satu
> negara pada dataset. Karena itu UI menguncinya sebagai metadata dan tidak
> membuat kombinasi negara/kategori yang tidak pernah ada.

**Best model berdasarkan RMSE:** **SVM** (RMSE = 0.0409, R² = 0.9994).
Hyper-parameter terpilih: `kernel=poly, C=10, epsilon=0.01, gamma=auto`.

> Dataset bersifat sintetis. Model mempelajari pola generator data dan tidak
> boleh diklaim sebagai prediktor harga BBM dunia nyata. Input Brent di luar
> rentang historis tahun terpilih akan diberi warning pada UI.

---

## 🛠️ Tech stack

- **Python 3.11**
- **scikit-learn** — KNN, SVR, Random Forest + GridSearchCV / RandomizedSearchCV
- **pandas / numpy** — manipulasi data & feature engineering
- **matplotlib / seaborn** — visualisasi (DPI 150)
- **joblib** — serialisasi model & preprocessor
- **Gradio ≥ 4.0** — antarmuka web (deploy ke Hugging Face Spaces)

---

## 💻 Cara run lokal

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Latih model (membuat models/*.pkl, model_comparison.json, plots)
python train.py                # semua model
# python train.py --model knn  # hanya satu model (knn|svm|rf)

# 3. Jalankan web app
python app.py
```

> Jika `models/` belum berisi artefak saat `app.py` dijalankan,
> `app_startup.py` akan otomatis menjalankan training terlebih dahulu.

### Struktur project

```
fuel_price_predictor/
├── data/
│   ├── global_fuel_prices_2020_2026.csv
│   ├── plots/                      # di-generate saat training / startup
│   └── model_comparison.json       # dibuat saat training
├── models/                         # *.pkl dibuat saat training
├── src/
│   ├── preprocessing.py            # DataPreprocessor
│   ├── eda.py                      # plot EDA
│   └── models/
│       ├── knn_model.py · svm_model.py · rf_model.py
│       └── _common.py              # metrik & plot bersama
├── notebooks/model_analysis.ipynb  # analisis error mendalam
├── app.py · train.py · app_startup.py
├── requirements.txt · README.md · .gitignore
```

---

## 🙏 Credit

Tugas Besar mata kuliah **Dasar Ilmu Data** (semester 3). Dibangun dengan
scikit-learn & Gradio. Dataset *Global Fuel Prices 2020–2026* digunakan untuk
keperluan edukasi/akademik.
