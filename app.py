import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sales Prediction - CRISP-DM",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "model/sales_model.pkl"
DATA_PATH = "data/superstore.csv"

st.title("Sales Prediction Dashboard")
st.caption("Machine Learning UAS — CRISP-DM | Random Forest Regressor")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model belum ditemukan. Jalankan notebook/sales_random_forest_crisp_dm.ipynb "
        "sampai bagian Deployment terlebih dahulu."
    )
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.warning("Dataset belum ditemukan. Jalankan notebook terlebih dahulu.")
    df = pd.DataFrame()

# Dashboard overview
st.header("Dashboard Statistik")

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{len(df):,}")
    c2.metric("Total Sales", f"${df['Sales'].sum():,.2f}" if "Sales" in df else "-")
    c3.metric("Rata-rata Sales", f"${df['Sales'].mean():,.2f}" if "Sales" in df else "-")
    c4.metric("Total Profit", f"${df['Profit'].sum():,.2f}" if "Profit" in df else "-")

    left, right = st.columns(2)

    with left:
        st.subheader("Sales berdasarkan Category")
        category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(category_sales)

    with right:
        st.subheader("Sales berdasarkan Region")
        region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        st.bar_chart(region_sales)

st.divider()

# Prediction
st.header("Prediksi Sales")
st.write("Masukkan karakteristik transaksi untuk mendapatkan estimasi Sales.")

col1, col2, col3 = st.columns(3)

with col1:
    ship_mode = st.selectbox(
        "Ship Mode",
        ["Standard Class", "Second Class", "First Class", "Same Day"]
    )
    segment = st.selectbox(
        "Segment",
        ["Consumer", "Corporate", "Home Office"]
    )
    region = st.selectbox(
        "Region",
        ["West", "East", "Central", "South"]
    )
    category = st.selectbox(
        "Category",
        ["Office Supplies", "Furniture", "Technology"]
    )

with col2:
    sub_category = st.selectbox(
        "Sub-Category",
        [
            "Paper", "Binders", "Art", "Storage", "Appliances",
            "Labels", "Envelopes", "Fasteners", "Supplies",
            "Chairs", "Tables", "Bookcases", "Furnishings",
            "Phones", "Accessories", "Machines", "Copiers"
        ]
    )
    quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2)
    discount = st.slider("Discount", 0.0, 0.8, 0.1, 0.05)

with col3:
    order_year = st.number_input("Tahun", min_value=2014, max_value=2030, value=2018)
    order_month = st.number_input("Bulan", min_value=1, max_value=12, value=6)
    order_day = st.number_input("Tanggal", min_value=1, max_value=31, value=15)
    order_weekday = st.selectbox(
        "Hari dalam minggu",
        list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ][x]
    )

if st.button("Prediksi Sales", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "ship_mode": ship_mode,
        "segment": segment,
        "region": region,
        "category": category,
        "sub_category": sub_category,
        "quantity": quantity,
        "discount": discount,
        "order_year": order_year,
        "order_month": order_month,
        "order_day": order_day,
        "order_weekday": order_weekday
    }])

    prediction = float(model.predict(input_data)[0])

    st.success(f"Estimasi Sales: **${prediction:,.2f}**")
    st.dataframe(input_data, use_container_width=True)

st.divider()

st.header("Tentang Model")
st.write(
    "Model yang digunakan adalah Random Forest Regressor. "
    "Model telah disimpan dalam format Pickle dan dimuat kembali oleh aplikasi "
    "tanpa melakukan training ulang."
)

st.info(
    "Catatan: jalankan notebook sekali untuk mengunduh dataset, melatih model, "
    "menghasilkan sales_model.pkl, dan menyiapkan data dashboard."
)
