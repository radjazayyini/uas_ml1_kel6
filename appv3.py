import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Forecast Wisatawan - ARIMA",
    layout="centered"
)

st.title("📈 Aplikasi Forecast Jumlah Wisatawan")
st.write("Aplikasi ini menampilkan EDA dan peramalan jumlah wisatawan menggunakan metode ARIMA.")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("model_arima_wisman.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("datasetfinal.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    return df

df = load_data()
ts = df['wisatawan']

# ============================================================
# SIDEBAR MENU
# ============================================================
menu = st.sidebar.radio(
    "📌 Menu",
    ("Beranda", "EDA", "Forecast ARIMA")
)

# ============================================================
# HALAMAN BERANDA
# ============================================================
if menu == "Beranda":
    st.subheader("🏠 Beranda")
    st.markdown("""
    **Aplikasi ini digunakan untuk:**
    - Eksplorasi data jumlah wisatawan (EDA)
    - Peramalan jumlah wisatawan menggunakan ARIMA
    - Visualisasi hasil forecasting
    
    Model telah dilatih sebelumnya dan disimpan dalam format `.pkl`.
    """)

# ============================================================
# HALAMAN EDA
# ============================================================
elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # Statistik deskriptif
    st.markdown("### 📌 Statistik Deskriptif")
    st.write(df.describe())

    # Plot time series
    st.markdown("### 📈 Grafik Time Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Wisatawan")
    st.pyplot(fig)

    # Rolling mean
    st.markdown("### 🔄 Rolling Mean (12 Bulan)")
    rolling_mean = ts.rolling(window=12).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, label="Data Aktual")
    ax.plot(rolling_mean, label="Rolling Mean", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Histogram
    st.markdown("### 📊 Distribusi Data")
    fig, ax = plt.subplots()
    ax.hist(ts.dropna(), bins=20)
    ax.set_xlabel("Jumlah Wisatawan")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # Boxplot
    st.markdown("### 📦 Boxplot (Outlier)")
    fig, ax = plt.subplots()
    ax.boxplot(ts.dropna(), vert=False)
    ax.set_xlabel("Jumlah Wisatawan")
    st.pyplot(fig)

    # Pola musiman
    st.markdown("### 🗓️ Pola Musiman Tahunan")
    df_seasonal = df.copy()
    df_seasonal['Year'] = df_seasonal.index.year
    df_seasonal['Month'] = df_seasonal.index.month

    seasonal_avg = df_seasonal.groupby('Month')['wisatawan'].mean()

    fig, ax = plt.subplots()
    ax.plot(seasonal_avg)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata Wisatawan")
    st.pyplot(fig)

# ============================================================
# HALAMAN FORECAST
# ============================================================
elif menu == "Forecast ARIMA":
    st.subheader("🔮 Forecast Jumlah Wisatawan")

    n_forecast = st.slider(
        "Jumlah periode forecast (bulan)",
        min_value=1,
        max_value=24,
        value=12
    )

    forecast = model.get_forecast(steps=n_forecast)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Grafik forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, label="Data Aktual")
    ax.plot(mean_forecast, label="Forecast", linestyle="--")
    ax.fill_between(
        mean_forecast.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3,
        label="Confidence Interval"
    )
    ax.legend()
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Wisatawan")
    st.pyplot(fig)

    # Tabel forecast
    st.markdown("### 📋 Tabel Hasil Forecast")
    forecast_df = pd.DataFrame({
        "Forecast": mean_forecast,
        "Lower CI": conf_int.iloc[:, 0],
        "Upper CI": conf_int.iloc[:, 1]
    })
    st.dataframe(forecast_df)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Aplikasi Forecast Wisatawan | ARIMA | Streamlit")
