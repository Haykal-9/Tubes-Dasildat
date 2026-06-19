# Brief: Perbaikan Model Prediksi Harga BBM

Catatan buat Claude Code. Ini ringkasan diagnosis dari project ML regresi prediksi harga BBM (KNN, SVM, Random Forest) dengan UI Gradio yang dideploy ke Hugging Face Spaces. Modelnya ngasih hasil yang kelihatan kayak anomali, dan di bawah ini penjelasan kenapa plus apa yang harus diperbaiki.

## Konteks dataset

File: `global_fuel_prices_2020_2026.csv`

- 27468 baris, 84 negara, data mingguan dari 2020-01 sampai 2026-04 (327 baris per negara)
- Kolom: `date`, `country`, `region`, `income_level`, `subsidy_level`, `petrol_usd_liter`, `diesel_usd_liter`, `lpg_usd_liter`, `brent_crude_usd`, `tax_percentage`
- Ada 3 target paralel: `petrol_usd_liter`, `diesel_usd_liter`, `lpg_usd_liter`

## Fakta penting dari data

- Ketiga harga fuel (petrol, diesel, lpg) korelasinya 0.999 satu sama lain. Jadi kalau dua di antaranya dipakai jadi fitur buat nebak yang ketiga, itu kebocoran data dan R² langsung melonjak ke 0.998 secara palsu.
- `country`, `region`, `income_level`, `subsidy_level` itu terkunci satu-satu per negara. Nol negara yang labelnya berubah. Artinya `country` udah otomatis mengandung info region, income, dan subsidy. Bikin keempatnya jadi kontrol independen di UI itu salah secara konsep dan menghasilkan kombinasi yang ga pernah eksis di data.
- 96% variasi harga itu murni karena beda negara, bukan waktu atau brent. Jadi model apapun gampang dapet akurasi tinggi cuma dengan ngapalin "negara X = harga Y". Akurasi 95%+ di sini menipu, bukan tanda model bagus.
- Brent crude di data ga realistis (sintetis). Range-nya cuma 47.97 sampai 130, naik tiap tahun (2020: 48-99, 2026: 118-130). Brent serendah 80 cuma ada di sekitar 2020. Pajak juga loncat-loncat tiap minggu (misal 59% lalu 27% lalu 62% di negara yang sama), yang ga masuk akal di dunia nyata.
- Contoh Algeria: selalu Africa, Middle income, Very High subsidy. Harga petrol rata-rata 0.141 USD/L (rentang 0.034 sampai 0.219). Prediksi model buat Algeria keluar di 0.136, yang sebenernya bener. Itu cuma kelihatan aneh karena dibandingin sama rata-rata global (2.28) atau Africa (1.55), padahal Algeria emang disubsidi habis-habisan.

Rata-rata petrol per region buat patokan: Middle East 1.23, South America 1.43, Africa 1.55, North America 2.06, Asia 2.14, Oceania 3.66, Europe 3.70. Global 2.28.

## Akar masalah

1. Akurasi tinggi itu palsu karena soalnya kelewat gampang (harga ditentuin hampir sepenuhnya sama negara) dan kemungkinan split data-nya random, jadi negara yang sama bocor ke train dan test sekaligus.
2. Prediksi kelihatan "jauh dari rata-rata" karena dibandingin ke patokan yang salah (global/region), bukan ke histori negaranya sendiri.
3. Input brent di luar range training (misal 80 buat 2026) bikin KNN ekstrapolasi ke kombinasi yang ga pernah dilihat, dan dia cuma balikin rata-rata tetangga terdekat.
4. Kontrol subsidy/income di UI praktis ga ngefek karena `country` mendominasi tetangga KNN, dan kombinasi kayak "Algeria + Medium subsidy" itu impossible.
5. Risiko leakage dari diesel dan lpg yang dipakai jadi fitur, plus KNN/SVM kemungkinan jalan tanpa scaling.

## Yang harus diperbaiki

- Buang `diesel_usd_liter` dan `lpg_usd_liter` dari daftar fitur kalau targetnya petrol. Perlakukan ketiganya sebagai target terpisah, bukan fitur buat satu sama lain.
- Pakai one-hot encoding buat kolom kategorikal, jangan LabelEncoder integer karena itu bikin jarak ordinal palsu yang ngerusak KNN dan SVM.
- Bungkus StandardScaler dalam sklearn Pipeline buat KNN dan SVM. Tanpa scaling R² turun ke 0.825, dengan scaling 0.998, karena brent dan year mendominasi perhitungan jarak. Random Forest ga butuh scaling tapi aman kalau ikut dalam pipeline terpisah.
- Ganti split jadi berbasis waktu, bukan random. Misal train pakai 2020 sampai 2025, test pakai 2026, biar test set bener-bener data baru.
- Laporkan MAE dan RMSE dalam USD/liter, jangan cuma R² atau "akurasi". Tambahin juga error relatif terhadap skala harga.
- UI pakai mode negara:
  - Sediakan dropdown `country` berisi 84 negara dari data.
  - Begitu negara dipilih, `region`, `income_level`, dan `subsidy_level` otomatis terisi dari lookup yang dibikin dari data (`df.groupby('country')[['region','income_level','subsidy_level']].first()`). Tampilkan ketiganya sebagai info read-only, bukan kontrol yang bisa diubah user.
  - Kontrol yang tetap bisa diatur user cuma `brent_crude_usd`, `tax_percentage`, `year`, dan `month`.
  - Buang semua slider atau dropdown independen buat region, income, dan subsidy. Itu sumber kombinasi impossible yang bikin prediksi ngaco.
  - Buat training model, `country` (one-hot) jadi fitur kategorikal utama. region/income/subsidy boleh ikut tapi sebenernya redundant karena nilainya udah ditentukan sama country, jadi sifatnya opsional.
- Clamp atau kasih warning kalau input brent di luar range realistis buat tahun yang dipilih, biar user ga minta prediksi yang sebenernya ekstrapolasi.
- Buat sanity check prediksi, banding ke rentang histori negara yang dipilih, bukan ke rata-rata global atau region.

## Catatan tambahan

Dataset ini sintetis, jadi yang dipelajari model itu pola generator datanya, bukan ekonomi harga BBM dunia yang sebenernya. Ga masalah buat tugas kuliah, tapi jangan diklaim sebagai prediktor harga BBM riil.
