#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Data

# In[2]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning  
warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='statsmodels')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import statsmodels.api as smt

# Mengimpor Data Interaksi
data_interaksi = pd.read_csv('FORM INTERAKSI-KOTAFM SURABAYA.csv')
data_interaksi


# ## 2. Preprocessing Data

# In[4]:


data_interaksi['Tanggal Siaran'] = pd.to_datetime(data_interaksi['Tanggal Siaran'], format='%d/%m/%Y')

cols_to_convert = ['WA', 'TELP', 'SMS', 'INSTAGRAM STORIES', 'TWITTER', 'FB']
for col in cols_to_convert:
    data_interaksi[col] = data_interaksi[col].fillna(0).astype(int)


# In[5]:


# Mengelompokkan data berdasarkan tanggal dan menjumlahkan interaksi
data_grouped = data_interaksi.groupby('Tanggal Siaran').agg({'WA': 'sum', 'TELP': 'sum', 'SMS': 'sum', 
                                                             'INSTAGRAM STORIES': 'sum', 'TWITTER': 'sum', 'FB': 'sum'}).reset_index()

# Menambahkan kolom total interaksi
data_grouped['Total Interaksi'] = data_grouped[['WA', 'TELP', 'SMS', 'INSTAGRAM STORIES', 'TWITTER', 'FB']].sum(axis=1)
data_grouped


# In[6]:


data_grouped.info()


# In[7]:


data_grouped.describe()


# ## 3. EDA

# In[9]:


# Menggunakan kolom 'Tanggal Siaran' sebagai indeks datetime
data_grouped['Tanggal Siaran'] = pd.to_datetime(data_grouped['Tanggal Siaran'])
data_grouped.set_index('Tanggal Siaran', inplace=True)


# In[10]:


# Menambahkan kolom hari dengan nama hari dalam bahasa Inggris
data_interaksi['Hari'] = data_interaksi['Tanggal Siaran'].dt.day_name()

# Mapping nama hari dari bahasa Inggris ke bahasa Indonesia
hari_mapping = {
    'Monday': 'Senin',
    'Tuesday': 'Selasa',
    'Wednesday': 'Rabu',
    'Thursday': 'Kamis',
    'Friday': 'Jumat',
    'Saturday': 'Sabtu',
    'Sunday': 'Minggu'
}
data_interaksi['Hari'] = data_interaksi['Hari'].map(hari_mapping)

# Mengelompokkan data berdasarkan hari dan menghitung total interaksi
total_interaksi_per_hari = data_interaksi.groupby('Hari')[['WA', 'TELP', 'SMS', 'INSTAGRAM STORIES', 'TWITTER', 'FB']].sum()

# Menambahkan kolom 'Total Interaksi' sebagai jumlah dari semua platform
total_interaksi_per_hari['Total Interaksi'] = total_interaksi_per_hari.sum(axis=1)

# Mengurutkan hari sesuai dengan urutan kalender dalam bahasa Indonesia
hari_urut = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
total_interaksi_per_hari = total_interaksi_per_hari.reindex(hari_urut)

# Visualisasi Total Interaksi Berdasarkan Hari dalam Bahasa Indonesia
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=total_interaksi_per_hari.index, y=total_interaksi_per_hari['Total Interaksi'], palette='viridis')
plt.title('Total Interaksi Berdasarkan Hari', fontsize=16, fontweight='bold')
plt.xlabel('Hari', fontsize=14)
plt.ylabel('Total Interaksi', fontsize=14)
plt.xticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[11]:


# Menghitung total jumlah interaksi per platform
platforms = ['WA', 'TELP', 'SMS', 'INSTAGRAM STORIES', 'TWITTER', 'FB']
total_interactions = data_grouped[platforms].sum()
print(total_interactions)

plt.figure(figsize=(12, 6))
sns.barplot(x=total_interactions.index, y=total_interactions.values, palette='Set2')
plt.title('Total Jumlah Interaksi per Platform', fontsize=18, fontweight='bold')
plt.xlabel('Platform', fontsize=14)
plt.ylabel('Total Jumlah Interaksi', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[12]:


# Visualisasi tren interaksi per platform
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

plt.figure(figsize=(14, 12))
for i, (platform, color) in enumerate(zip(platforms, colors), 1):
    plt.subplot(len(platforms), 1, i)
    plt.plot(data_grouped[platform], label=platform, color=color)
    plt.title(f'Trend Interaksi - {platform}', fontsize=12, fontweight='bold')
    plt.xlabel('Tanggal Siaran', fontsize=10)
    plt.ylabel('Jumlah Interaksi', fontsize=10)
    plt.grid(alpha=0.5)
    plt.legend(loc="upper left")

plt.tight_layout()
plt.show()


# In[13]:


# Menambahkan kolom minggu ke dalam data
data_interaksi['Minggu'] = data_interaksi['Tanggal Siaran'].dt.isocalendar().week

# Mengelompokkan data berdasarkan minggu
total_interaksi_per_minggu = data_interaksi.groupby('Minggu')[['WA', 'TELP', 'SMS', 'INSTAGRAM STORIES', 'TWITTER', 'FB']].sum()

# Visualisasi pola interaksi mingguan
plt.figure(figsize=(12, 6))
sns.lineplot(data=total_interaksi_per_minggu.sum(axis=1), marker='o', label='Total Interaksi Mingguan')
plt.title('Pola Interaksi Mingguan', fontsize=16, fontweight='bold')
plt.xlabel('Minggu ke-', fontsize=14)
plt.ylabel('Total Interaksi', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ## 4. Model ARIMA

# In[15]:


# Menggunakan kembali kolom Tanggal Siaran untuk Plotly
figure = px.line(
    data_grouped.reset_index(),  
    x="Tanggal Siaran",
    y="Total Interaksi",
    title="Total Interaksi Kota FM"
)
figure.show()


# In[16]:


# Fungsi untuk mengevaluasi stasioneritas varians menggunakan Box-Cox
def evaluate_variance_stationarity(timeseries):
    # Menghapus nilai nol atau negatif karena Box-Cox hanya menerima nilai positif
    timeseries = timeseries[timeseries > 0]

    # Melakukan transformasi Box-Cox
    transformed_data, lambda_value = boxcox(timeseries)

    # Menampilkan hasil transformasi
    print("Hasil Transformasi Box-Cox:")
    print(f"Nilai Lambda (λ): {lambda_value:.2f}")
    print(f"Rounded Value (λ): {round(lambda_value)}")

    # Interpretasi hasil dengan hipotesis H₀ dan H₁
    print("\nInterpretasi Hasil:")
    print("Hipotesis Uji:")
    print("H₀: λ ≠ 1 (data tidak memiliki varians yang stabil)")
    print("H₁: λ = 1 (data memiliki varians yang stabil)")

    if round(lambda_value) == 1:
        print("\nKesimpulan: H₀ ditolak. Data sudah stasioner dalam varians (memiliki varians yang stabil).")
    else:
        print("\nKesimpulan: H₀ gagal ditolak. Data belum stasioner dalam varians. Transformasi tambahan mungkin diperlukan.")
        print("Proses transformasi lebih lanjut bisa dilakukan untuk mencapai λ ≈ 1.")

# Menggunakan kolom 'Total Interaksi' dari dataset
evaluate_variance_stationarity(data_grouped['Total Interaksi'])


# In[17]:


# Fungsi untuk uji Dickey-Fuller dengan kesimpulan
def dickey_fuller_test(timeseries):
    # Melakukan uji Dickey-Fuller
    print("Hasil Uji Dickey-Fuller:")
    dftest = adfuller(timeseries.dropna(), autolag='AIC')  
    dfoutput = pd.Series(dftest[0:4], index=['Statistik Uji', 'P-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    # Menyusun hasil interpretasi berdasarkan perbandingan statistik dengan nilai kritis
    print("\nInterpretasi Hasil:")
    print("Hipotesis Uji:")
    print("H₀: ϕ = 0 (deret waktu tidak stasioner)")
    print("H₁: ϕ < 0 (deret waktu stasioner)")
    
    # Membandingkan statistik uji dengan nilai kritis pada berbagai level signifikansi
    critical_values = dftest[4]  # Nilai kritis untuk 1%, 5%, dan 10%
    test_statistic = dftest[0]  # Nilai statistik uji ADF
    
    # Interpretasi berdasarkan perbandingan statistik uji dengan nilai kritis
    if test_statistic < critical_values['1%']:
        print("\nKeputusan: H₀ ditolak pada level signifikansi 1%. Data stasioner.")
    elif test_statistic < critical_values['5%']:
        print("\nKeputusan: H₀ ditolak pada level signifikansi 5%. Data stasioner.")
    elif test_statistic < critical_values['10%']:
        print("\nKeputusan: H₀ ditolak pada level signifikansi 10%. Data stasioner.")
    else:
        print("\nKeputusan: H₀ gagal ditolak. Data tidak stasioner.")

# Melakukan uji Dickey-Fuller pada 'Total Interaksi'
dickey_fuller_test(data_grouped['Total Interaksi'])


# In[18]:


# Menentukan model ARIMA dengan differencing (d = 0)
# Plot ACF dan PACF
plt.figure(figsize=(12, 6))

plt.subplot(121)
plot_acf(data_grouped['Total Interaksi'], lags=20, ax=plt.gca())

plt.subplot(122)
plot_pacf(data_grouped['Total Interaksi'], lags=20, ax=plt.gca())

plt.show()


# In[19]:


# Data frame untuk menyimpan hasil estimasi parameter
results = []

# Daftar model ARIMA yang ingin diuji (termasuk p=0 dan q=0 yang dikombinasikan dengan p dan q lainnya)
models = [
    (0, 0, 0), 
    (0, 0, 3),  
    (0, 0, 4), 
    (0, 0, 6), 
    (3, 0, 0), 
    (3, 0, 3),  
    (3, 0, 4), 
    (3, 0, 6), 
    (6, 0, 0),  
    (6, 0, 3),  
    (6, 0, 4),  
    (6, 0, 6)   
]

# Loop untuk mengestimasi setiap model ARIMA dan menyimpan hasilnya
for order in models:
    model = ARIMA(data_grouped['Total Interaksi'], order=order)
    model_fitted = model.fit()
    
    # Menyimpan hasil parameter dan p-value hanya jika p-value < 0.05
    for param, coef, p_value in zip(model_fitted.params.index, model_fitted.params.values, model_fitted.pvalues.values):
        if p_value < 0.05:
            results.append({
                'Model': f"ARIMA{order}",
                'Parameter': param,
                'Koefisien': coef,
                'p-value': p_value,
                'Keterangan': 'Signifikan'
            })

# Membuat DataFrame dari hasil estimasi
df_results = pd.DataFrame(results)
df_results


# In[27]:


# Fungsi untuk uji diagnostik White Noise dan Normalitas
def diagnostic_checks(model_fitted):
    residuals = model_fitted.resid  # Mengambil residual dari model
    
    # Uji Ljung-Box untuk white noise (p-value)
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].iloc[0]  # Akses dengan iloc untuk posisi pertama
    
    # Uji Shapiro-Wilk untuk normalitas (p-value)
    shapiro_test = shapiro(residuals)
    shapiro_pvalue = shapiro_test.pvalue
    
    return lb_pvalue, shapiro_pvalue

# Daftar model ARIMA yang signifikan berdasarkan hasil estimasi sebelumnya
models_signifikan = [
    (0, 0, 0),
    (0, 0, 3),
    (0, 0, 4),
    (0, 0, 6),
    (3, 0, 0),
    (3, 0, 3),
    (3, 0, 4),
    (3, 0, 6),
    (6, 0, 0),
    (6, 0, 3),
    (6, 0, 4),
    (6, 0, 6)
]

# Data frame untuk menyimpan hasil uji diagnostik
diagnostic_results = []

# Loop untuk menguji setiap model ARIMA yang signifikan
for order in models_signifikan:
    # Membuat dan mengestimasi model ARIMA
    model = ARIMA(data_grouped['Total Interaksi'], order=order)
    model_fitted = model.fit()

    # Melakukan uji Ljung-Box (White Noise) dan Shapiro-Wilk (Normalitas) pada model
    lb_pvalue, shapiro_pvalue = diagnostic_checks(model_fitted)
    
    # Memastikan bahwa model memenuhi asumsi white noise dan normalitas
    if lb_pvalue > 0.05 and shapiro_pvalue > 0.05:
        diagnostic_results.append({
            'Model': f"ARIMA{order}",
            'White Noise p-value': lb_pvalue,
            'Normalitas p-value': shapiro_pvalue
        })

# Membuat DataFrame dari hasil diagnostik untuk model yang memenuhi asumsi
df_diagnostic_results = pd.DataFrame(diagnostic_results)

# Menampilkan hasil
print("\nHasil Diagnostik untuk Model yang Memenuhi Asumsi White Noise dan Normalitas (p-value > 0.05):")
df_diagnostic_results


# In[28]:


# Data frame untuk menyimpan hasil AIC
aic_results = []

# Daftar model ARIMA yang memenuhi asumsi white noise dan normalitas berdasarkan hasil sebelumnya
valid_models = [
    (0, 0, 4),
    (0, 0, 6),
    (3, 0, 0),
    (6, 0, 0)
]

# Loop untuk menguji setiap model ARIMA
for order in valid_models:
    # Membuat dan mengestimasi model ARIMA
    model = ARIMA(data_grouped['Total Interaksi'], order=order)
    model_fitted = model.fit()

    # Mengambil AIC dari model yang sudah di-fit
    aic = model_fitted.aic

    # Menyimpan hasil AIC ke dalam list
    aic_results.append({
        'Model': f"ARIMA{order}",
        'AIC': aic
    })

# Membuat DataFrame dari hasil AIC
df_aic_results = pd.DataFrame(aic_results)

# Menampilkan hasil AIC untuk model-model yang valid
print("\nHasil AIC untuk Model yang Memenuhi Asumsi White Noise dan Normalitas:")
df_aic_results


# In[29]:


# Menentukan model dengan AIC terkecil (model terbaik)
best_model = df_aic_results.loc[df_aic_results['AIC'].idxmin()]

print("\nModel Terbaik Berdasarkan AIC:")
best_model


# In[72]:


# Silakan sesuaikan sesuai data yang Anda miliki
data = data_grouped['Total Interaksi']

# Model terbaik (misalnya ARIMA(6,0,0))
best_model_order = (6, 0, 0)

# Membuat dan mengestimasi model terbaik
model = ARIMA(data, order=best_model_order)
model_fitted = model.fit()

# Menyimpan parameter model
params = model_fitted.params

# Menampilkan parameter model
print("\nParameter ARIMA (6,0,0):")
for param, value in params.items():
    print(f"{param}: {value:.6f}")


# In[30]:


# Membuat dan mengestimasi model ARIMA(6, 0, 0)
best_order = (6, 0, 0)
model = ARIMA(data_grouped['Total Interaksi'], order=best_order)
model_fitted = model.fit()

# Melakukan prediksi untuk data yang sudah ada (in-sample forecast)
predictions = model_fitted.predict(start=0, end=len(data_grouped['Total Interaksi']) - 1, typ='levels')

# Menghitung MAPE
actual = data_grouped['Total Interaksi']
mape = np.mean(np.abs((actual - predictions) / actual)) * 100

# Menampilkan hasil MAPE
print(f"MAPE untuk model ARIMA{best_order}: {mape:.2f}%")


# In[31]:


# Membuat dan mengestimasi model ARIMA(6, 0, 0)
best_order = (6, 0, 0)
model = ARIMA(data_grouped['Total Interaksi'], order=best_order)
model_fitted = model.fit()

# Melakukan prediksi untuk 14 hari ke depan
forecast_steps = 14  
forecast_values = model_fitted.forecast(steps=forecast_steps)

# Membulatkan hasil prediksi menjadi integer
forecast_values_int = np.round(forecast_values).astype(int)

# Membuat tanggal prediksi (tanggal berikutnya dari data terakhir)
last_date = pd.to_datetime(data_grouped.index[-1])
predicted_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_steps+1)]

# Membuat DataFrame hasil prediksi
df_forecast = pd.DataFrame({
    'Tanggal Prediksi': predicted_dates,
    'Prediksi Total Interaksi': forecast_values_int
})

# Menampilkan hasil prediksi dalam tabel
df_forecast


# In[56]:


# Membuat plot untuk data historis dan prediksi
plt.figure(figsize=(10, 6))

# Plot data historis
plt.plot(data_grouped.index, data_grouped['Total Interaksi'], label='Data Historis', color='blue')

# Plot hasil prediksi dengan garis lurus
plt.plot(df_forecast['Tanggal Prediksi'], df_forecast['Prediksi Total Interaksi'], label='Prediksi 14 Hari', color='red')

# Menambahkan label dan judul
plt.title('Prediksi Total Interaksi (ARIMA(6, 0, 0)) untuk 14 Hari Ke Depan')
plt.xlabel('Tanggal')
plt.ylabel('Total Interaksi')
plt.legend()

# Menampilkan grid dan plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[68]:


import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# Membuat dan mengestimasi model ARIMA(6, 0, 0) berdasarkan data historis
best_order = (6, 0, 0)
model = ARIMA(data_grouped['Total Interaksi'], order=best_order)
model_fitted = model.fit()

# Melakukan prediksi untuk data aktual (in-sample forecast)
actual_predictions = model_fitted.predict(start=0, end=len(data_grouped['Total Interaksi']) - 1, typ='levels')

# Melakukan prediksi untuk 14 hari ke depan
forecast_steps = 14  
forecast_values = model_fitted.forecast(steps=forecast_steps)

# Membulatkan hasil prediksi menjadi integer
forecast_values_int = np.round(forecast_values).astype(int)

# Membuat tanggal prediksi (tanggal berikutnya dari data terakhir)
last_date = pd.to_datetime(data_grouped.index[-1])
predicted_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_steps+1)]

# Membuat DataFrame hasil prediksi
df_forecast = pd.DataFrame({
    'Tanggal Prediksi': predicted_dates,
    'Prediksi Total Interaksi': forecast_values_int
})

# Membuat plot untuk data historis dan prediksi
plt.figure(figsize=(10, 6))

# Plot data historis
plt.plot(data_grouped.index, data_grouped['Total Interaksi'], label='Data Historis', color='blue')

# Plot prediksi data aktual
plt.plot(data_grouped.index, actual_predictions, label='Prediksi Data Aktual', color='green')

# Plot hasil prediksi 14 hari ke depan
plt.plot(df_forecast['Tanggal Prediksi'], df_forecast['Prediksi Total Interaksi'], label='Prediksi 14 Hari', color='red')

# Menambahkan label dan judul
plt.title('Prediksi Total Interaksi (ARIMA(6, 0, 0)) untuk 14 Hari Ke Depan')
plt.xlabel('Tanggal')
plt.ylabel('Total Interaksi')
plt.legend()

# Menampilkan grid dan plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




