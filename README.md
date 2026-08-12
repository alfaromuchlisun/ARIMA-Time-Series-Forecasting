# Time Series Analysis & Forecasting using ARIMA (Kota FM Surabaya)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Project ini melakukan analisis deret waktu (*time series analysis*) dan pemodelan peramalan (*forecasting*) total interaksi pendengar pada radio **Kota FM Surabaya** menggunakan metode **ARIMA (Autoregressive Integrated Moving Average)**.

---

## Features & Workflow

1. **Data Preprocessing & Aggregation:**
   - Pembersihan data interaksi lintas kanal (WhatsApp, Telepon, SMS, Instagram Stories, Twitter, Facebook).
   - Agregasi harian total interaksi.
2. **Exploratory Data Analysis (EDA):**
   - Pola interaksi berdasarkan hari siaran.
   - Distribusi total interaksi per platform.
   - Tren interaksi mingguan.
3. **Time Series Diagnostics:**
   - **Stasioneritas Varians:** Evaluasi transformasi Box-Cox ($\lambda$).
   - **Stasioneritas Rataan:** Uji Augmented Dickey-Fuller (ADF).
   - Identification orde $p$ dan $q$ melalui grafik ACF & PACF.
4. **Model Selection & Diagnostics:**
   - Estimasi dan signifikansi parameter berbagai kombinasi ARIMA.
   - Diagnostic check: Uji White Noise (Ljung-Box) & Uji Normalitas Residual (Shapiro-Wilk).
   - Pemilihan model terbaik berdasarkan nilai **Akaike Information Criterion (AIC)** terendah.
5. **Forecasting & Evaluation:**
   - Evaluasi performa in-sample menggunakan **MAPE (Mean Absolute Percentage Error)**.
   - Out-of-sample forecasting 14 hari ke depan.

---

## Hasil Pemodelan

- **Model Terbaik:** ARIMA(6, 0, 0)
- **Evaluasi Model:** 
  - Residual memenuhi asumsi White Noise ($p\text{-value} > 0.05$) dan Normalitas ($p\text{-value} > 0.05$).
  - Performa akurasi dinilai menggunakan MAPE.
- **Forecasting:** Prediksi tren total interaksi harian untuk 14 hari ke depan.


cd ARIMA-Time-Series-Forecasting
