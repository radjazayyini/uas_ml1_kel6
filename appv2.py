import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(
    page_title="Forecasting Wisatawan",
    layout="wide"
)


@st.cache_resource
def load_model():
    with open("model_arima_wisman2.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_data():
    df = pd.read_csv("datasetfinal.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.asfreq("MS")
    return df

model = load_model()
df = load_data()
ts = df["wisatawan"]


st.sidebar.title("⚙️ Panel Forecast")

steps = st.sidebar.slider(
    "Periode Prediksi (bulan)",
    min_value=1,
    max_value=36,
    value=12
)

show_ci = st.sidebar.checkbox("Tampilkan Confidence Interval", value=True)


st.markdown(
    """
    ## 📈 Forecasting Jumlah Wisatawan  
    *Peramalan berbasis model ARIMA untuk mendukung pengambilan keputusan*
    """
)


forecast = model.get_forecast(steps=steps)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()


col1, col2, col3, col4 = st.columns(4)

col1.metric("📅 Data Terakhir", ts.index[-1].strftime("%Y-%m"))
col2.metric("📌 Nilai Terakhir", f"{ts.iloc[-1]:,.0f}")
col3.metric("📊 Rata-rata Forecast", f"{mean_forecast.mean():,.0f}")
col4.metric("📈 Tren", "Stabil" if mean_forecast.diff().mean() < 1000 else "Meningkat")


st.subheader("📉 Grafik Data Aktual & Prediksi")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(ts.index, ts, label="Data Aktual")
ax.axvline(ts.index[-1], linestyle="--", linewidth=1)
ax.plot(mean_forecast.index, mean_forecast, label="Forecast")

if show_ci:
    ax.fill_between(
        mean_forecast.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3,
        label="Confidence Interval"
    )

ax.set_title("Forecast Jumlah Wisatawan (2008 – Prediksi)")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Wisatawan")
ax.legend()

st.pyplot(fig)


st.subheader("📋 Tabel Hasil Prediksi")

forecast_df = pd.DataFrame({
    "Tanggal": mean_forecast.index,
    "Prediksi": mean_forecast.values,
    "CI Bawah": conf_int.iloc[:, 0].values,
    "CI Atas": conf_int.iloc[:, 1].values
})

st.dataframe(forecast_df, use_container_width=True)

st.download_button(
    label="⬇️ Download CSV",
    data=forecast_df.to_csv(index=False),
    file_name="hasil_forecast.csv",
    mime="text/csv"
)


st.subheader("📐 Evaluasi Model (In-Sample)")

pred_in = model.predict(start=1, end=len(ts)-1, typ="levels")
actual = ts[1:]

mae = mean_absolute_error(actual, pred_in)
rmse = np.sqrt(mean_squared_error(actual, pred_in))
mape = np.mean(np.abs((actual - pred_in) / actual)) * 100
accuracy = 100 - mape

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("MAPE", f"{mape:.2f}%")
col4.metric("Akurasi", f"{accuracy:.2f}%")


st.subheader("🧠 Insight & Rekomendasi")

if accuracy > 85:
    kualitas = "baik"
else:
    kualitas = "perlu peningkatan"

st.markdown(
    f"""
    **Insight Model:**  
    Model ARIMA menunjukkan performa **{kualitas}** dengan tingkat akurasi sebesar **{accuracy:.2f}%**.
    Hasil peramalan menunjukkan pola **relatif stabil** dengan tingkat ketidakpastian yang meningkat
    seiring bertambahnya horizon prediksi.

    **Rekomendasi:**  
    Forecast ini **layak digunakan untuk perencanaan jangka pendek (≤ 1 tahun)**.  
    Untuk jangka panjang disarankan menggunakan model dengan komponen musiman (SARIMA).
    """
)
