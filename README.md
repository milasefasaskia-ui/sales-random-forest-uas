# Sales Random Forest — UAS Machine Learning

## Implementasi CRISP-DM

Project ini dibuat untuk memenuhi tugas UAS Machine Learning dengan tahapan:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

### Topik
**Prediksi nilai Sales transaksi retail menggunakan Random Forest Regressor.**

### Dataset
Menggunakan **Sample Superstore / Superstore Sales Dataset**. Dataset publik ini berisi transaksi retail dengan atribut seperti Sales, Profit, Discount, Quantity, Category, Sub-Category, Region, Segment, dan informasi transaksi lainnya.

Sumber:
- Kaggle: https://www.kaggle.com/datasets/himanshuuike/superstore-sales-dataset
- Raw CSV publik: https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv

Notebook akan mengunduh data saat dijalankan dan menyimpannya ke `data/superstore.csv`.

## Struktur

```text
sales-random-forest/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── superstore.csv        # dibuat saat notebook dijalankan
├── model/
│   └── sales_model.pkl       # dibuat saat notebook selesai
└── notebook/
    └── sales_random_forest_crisp_dm.ipynb
```

## Instalasi

Buka terminal pada folder project:

```bash
pip install -r requirements.txt
```

## Training model

Buka Jupyter:

```bash
jupyter notebook
```

Kemudian buka:

```text
notebook/sales_random_forest_crisp_dm.ipynb
```

Jalankan semua cell dari atas sampai bawah.

Notebook akan:
- mengunduh dataset
- melakukan cleaning
- melakukan EDA
- melakukan feature engineering
- melakukan preprocessing
- membagi data training/testing
- melatih Random Forest
- menghitung MAE, RMSE, R²
- menyimpan `model/sales_model.pkl`

## Menjalankan dashboard

Setelah model berhasil dibuat:

```bash
streamlit run app.py
```

Dashboard akan membuka browser lokal.

## Evaluasi

Metrik yang digunakan:
- MAE
- RMSE
- R²

Nilai final harus diambil dari output notebook setelah training dijalankan.

## Catatan akademik

`Profit` tidak digunakan sebagai fitur utama untuk prediksi Sales karena Profit merupakan hasil finansial transaksi dan dapat menyebabkan target leakage dalam skenario prediksi.

Project ini menggunakan dataset publik untuk kebutuhan pembelajaran. Sumber dataset dicantumkan agar proses dapat direproduksi.

## Demo UAS

Urutan demonstrasi:
1. Jelaskan Business Understanding.
2. Tunjukkan dataset dan atribut.
3. Tunjukkan cleaning/preprocessing.
4. Tunjukkan training Random Forest.
5. Tunjukkan MAE, RMSE, R².
6. Tunjukkan file `sales_model.pkl`.
7. Jalankan `streamlit run app.py`.
8. Masukkan contoh transaksi.
9. Tunjukkan hasil prediksi di dashboard.
