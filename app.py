import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction Dashboard", page_icon="📊", layout="wide")

MODEL_PATH = "model/sales_model.pkl"
DATA_PATH = "data/superstore.csv"

st.title("Sales Prediction Dashboard")
st.caption("UAS Machine Learning — CRISP-DM | Random Forest Regressor")

if not os.path.exists(MODEL_PATH):
    st.error("File model/sales_model.pkl belum tersedia.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_PATH)
df["order_date"] = pd.to_datetime(df["order_date"])

c1,c2,c3,c4=st.columns(4)
c1.metric("Total Transaksi", f"{len(df):,}")
c2.metric("Total Sales", f"${df['sales'].sum():,.2f}")
c3.metric("Rata-rata Sales", f"${df['sales'].mean():,.2f}")
c4.metric("Total Quantity", f"{df['quantity'].sum():,}")

a,b=st.columns(2)
with a:
    st.subheader("Sales berdasarkan Kategori")
    st.bar_chart(df.groupby("category")["sales"].sum())
with b:
    st.subheader("Sales berdasarkan Region")
    st.bar_chart(df.groupby("region")["sales"].sum())

st.divider()
st.header("Prediksi Sales")

c1,c2,c3=st.columns(3)
with c1:
    ship_mode=st.selectbox("Ship Mode", sorted(df.ship_mode.unique()))
    segment=st.selectbox("Segment", sorted(df.segment.unique()))
    region=st.selectbox("Region", sorted(df.region.unique()))
    category=st.selectbox("Category", sorted(df.category.unique()))
with c2:
    sub_options=sorted(df[df.category==category].sub_category.unique())
    sub_category=st.selectbox("Sub-Category", sub_options)
    quantity=st.number_input("Quantity",1,100,2)
    discount=st.slider("Discount",0.0,0.8,0.1,0.05)
with c3:
    order_date=st.date_input("Order Date", value=df.order_date.iloc[-1].date())
    st.write("Hari:", order_date.strftime("%A"))

if st.button("Prediksi Sales", type="primary", use_container_width=True):
    d=pd.Timestamp(order_date)
    input_data=pd.DataFrame([{
        "ship_mode":ship_mode,
        "segment":segment,
        "region":region,
        "category":category,
        "sub_category":sub_category,
        "quantity":quantity,
        "discount":discount,
        "order_year":d.year,
        "order_month":d.month,
        "order_day":d.day,
        "order_weekday":d.dayofweek
    }])
    prediction=float(model.predict(input_data)[0])
    st.success(f"Estimasi Sales: **${prediction:,.2f}**")
    st.dataframe(input_data,use_container_width=True)

st.divider()
st.subheader("Evaluasi Model")
m1,m2,m3=st.columns(3)
m1.metric("MAE","245.47")
m2.metric("RMSE","453.17")
m3.metric("R²","0.1155")
st.caption("Metrik berasal dari hold-out test set 20% pada subset 84 transaksi publik yang disertakan.")
