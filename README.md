# Sales Random Forest — UAS Machine Learning

## CRISP-DM
Project ini menerapkan:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

### Topik
Prediksi nilai Sales transaksi retail menggunakan Random Forest Regressor.

### Dataset
Project menggunakan subset 84 transaksi dari **Sample Superstore / Superstore Sales Dataset**, dataset retail publik.

Sumber publik:
- Kaggle: https://www.kaggle.com/datasets/himanshuuike/superstore-sales-dataset
- GitHub: https://github.com/leonism/sample-superstore

Subset disertakan langsung pada `data/superstore.csv` agar aplikasi dapat berjalan tanpa download tambahan.

### Model
File `model/sales_model.pkl` sudah tersedia dan dapat dimuat langsung oleh aplikasi. Aplikasi tidak melakukan training ulang ketika dijalankan.

### Menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Notebook
Buka `notebook/sales_random_forest_crisp_dm.ipynb` untuk melihat cleaning, preprocessing, training, evaluasi, dan export model.

### Evaluasi
MAE: 245.47  
RMSE: 453.17  
R²: 0.1155

Catatan: subset yang disertakan berukuran kecil sehingga metrik adalah hasil prototype pembelajaran, bukan performa produksi.
