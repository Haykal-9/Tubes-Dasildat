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
faktor ekonomi & kebijakan (harga minyak Brent, pajak, tingkat subsidi, tingkat
pendapatan, region, dan waktu). Tiga model regresi — **KNN**, **SVM**, dan
**Random Forest** — dilatih, dibandingkan, dan disajikan lewat web app Gradio
yang siap di-deploy ke **Hugging Face Spaces**.

---

## 🚀 Cara pakai web app

1. Buka tab **Prediksi Harga BBM**.
2. Pilih **country**, **region**, **income level**, dan **subsidy level**.
3. Atur slider **Brent Crude (USD/barrel)**, **Tax (%)**, **Year**, dan pilih **Month**.
4. Pilih model (**KNN / SVM / Random Forest**) lalu tekan **Prediksi**.
5. Lihat harga prediksi, badge model, R² model, *confidence note*, serta
   tabel perbandingan terhadap rata-rata region & rata-rata global.

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

**Fitur yang digunakan:** `region` (One-Hot, drop_first), `income_level`
(ordinal), `subsidy_level` (ordinal), `country` (label encoding),
`brent_crude_usd` & `tax_percentage` (StandardScaler), serta `year` & `month`
yang diekstrak dari `date` (StandardScaler).

---

## 🤖 Tiga model & kapan dipilih

| Model | Tuning | Inti metode | Kapan dipilih |
|---|---|---|---|
| **KNN** | `GridSearchCV` (n_neighbors, weights, metric) | Rata-rata target dari *k* tetangga terdekat | Pola lokal, dataset tidak terlalu besar |
| **SVM (SVR)** | `GridSearchCV` (kernel, C, epsilon, gamma); tune di subset kecil → refit di subsample 10.000; kernel linier via `LinearSVR` | Margin ε + kernel untuk non-linieritas | Hubungan non-linier, dimensi fitur tinggi |
| **Random Forest** | `RandomizedSearchCV` (n_iter=30) | Ansambel decision tree (bagging) | Akurasi tinggi + *feature importance*, baseline kuat data tabular |

---

## 📈 Hasil perbandingan model

> Hasil di bawah dari training pada test set 5.494 baris (split 80:20,
> `random_state=42`). Nilai lengkap tersimpan di `data/model_comparison.json`.

| Model | MAE | RMSE | R² | MAPE (%) | Akurasi (R²) | Ketepatan (100−MAPE) |
|---|---|---|---|---|---|---|
| KNN | 0.0387 | 0.0565 | 0.9987 | 4.57 | 99.87% | 95.4% |
| SVM | 0.0382 | 0.0488 | 0.9990 | 10.67 | 99.90% | 89.3% |
| **Random Forest** 🏆 | **0.0250** | **0.0344** | **0.9995** | **4.70** | **99.95%** | **95.3%** |

> **Cara baca:** *MAE/RMSE* adalah **error** (USD/liter — makin kecil makin bagus,
> 0 = sempurna), **bukan** akurasi. *Akurasi (R²)* = proporsi variasi harga yang
> dijelaskan model, *Ketepatan (100−MAPE)* = rata-rata ketepatan prediksi. Dengan
> akurasi **89–99%**, ketiga model sudah jauh di atas target 80%.

> **Catatan tuning KNN:** karena `country` menentukan ~90% level harga, KNN paling
> akurat saat kolom `country` *diperkuat* (×10) sehingga tetangga selalu dari negara
> yang sama, lalu diinterpolasi pada fitur ekonomi/waktu. Justru men-`StandardScaler`
> semua fitur atau *target-encoding* `country` memperburuk KNN drastis
> (RMSE 0.0565 → 0.20 / 0.12) karena merusak pencocokan antar-negara. KNN sudah
> mendekati plafonnya; akurasi terbaik tetap di Random Forest.

**Best model:** **Random Forest** (RMSE terendah = 0.0344, R² = 0.9995).
Hyper-parameter terpilih: `n_estimators=300, max_depth=50, min_samples_split=5,
min_samples_leaf=1, max_features=None, bootstrap=False`.

> Catatan: ketiga model sangat akurat karena `country` (label-encoded) hampir
> sepenuhnya menentukan level harga tiap negara; Random Forest unggul dengan
> menangkap interaksi non-linier antar fitur tanpa perlu penskalaan.

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
