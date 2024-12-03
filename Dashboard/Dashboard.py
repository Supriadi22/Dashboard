import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import KMeans

# Load datasheet
hour_df = pd.read_csv('D:\Project Analisis Data\Bike-sharing-dataset\hour.csv')  # Arahkan dengan path file 'hour.csv'
day_df = pd.read_csv('D:\Project Analisis Data\Bike-sharing-dataset\day.csv')    # Arahkan dengan path file 'day.csv'

# Streamlit Title
st.title("Dashboard Analisis Sewa Sepeda")

# Menampilkan beberapa statistik dasar
st.header("Statistik Dasar")
st.subheader("Statistik Dataset Hour")
st.write(hour_df.describe())

# Visualisasi 1: Distribusi Jumlah Sewa Sepeda Berdasarkan Cuaca (Weathersit)
st.header("Distribusi Jumlah Sewa Sepeda Berdasarkan Cuaca")
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(data=hour_df, x='weathersit', y='cnt', ax=ax)
ax.set_title('Distribusi Jumlah Sewa Sepeda Berdasarkan Cuaca')
ax.set_xlabel('Kondisi Cuaca')
ax.set_ylabel('Jumlah Sewa Sepeda')
st.pyplot(fig)

# Visualisasi 2: Jumlah Sewa Sepeda Berdasarkan Jam
st.header("Jumlah Sewa Sepeda Berdasarkan Jam")
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(data=hour_df, x='hr', y='cnt', ax=ax, marker='o')
ax.set_title('Jumlah Sewa Sepeda Berdasarkan Jam')
ax.set_xlabel('Jam')
ax.set_ylabel('Jumlah Sewa Sepeda')
st.pyplot(fig)

# Filter interaktif: Memilih data berdasarkan suhu dan status hari libur
st.header("Hubungan Suhu dan Jumlah Sewa Sepeda (Hari Kerja vs Hari Libur)")

holiday_filter = st.selectbox('Pilih Hari', ['Hari Kerja', 'Hari Libur'])

# Filter data berdasarkan hari libur (holiday == 0 untuk hari kerja, holiday == 1 untuk hari libur)
if holiday_filter == 'Hari Kerja':
    filtered_df = hour_df[hour_df['holiday'] == 0]
else:
    filtered_df = hour_df[hour_df['holiday'] == 1]

# Visualisasi: Scatter plot suhu dan jumlah sewa sepeda
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=filtered_df, x='temp', y='cnt', hue='holiday', palette={0: 'blue', 1: 'red'}, ax=ax)
ax.set_title(f'Hubungan Suhu dan Jumlah Sewa Sepeda ({holiday_filter})')
ax.set_xlabel('Suhu (Temp)')
ax.set_ylabel('Jumlah Sewa Sepeda')
st.pyplot(fig)

# Statistik jumlah sewa berdasarkan filter
st.subheader(f"Statistik Jumlah Sewa Sepeda untuk {holiday_filter}")
st.write(filtered_df['cnt'].describe())

# Menampilkan tabel data terfilter
st.subheader(f"Tabel Data untuk {holiday_filter}")
st.write(filtered_df.head())

# Sample Data for RFM analysis
data = {
    'user_id': [1, 2, 3, 1, 2, 3],
    'transaction_date': ['2023-11-01', '2023-11-05', '2023-11-10', '2023-11-15', '2023-11-20', '2023-11-25'],
    'amount': [100, 200, 150, 50, 300, 400]
}

# DataFrame untuk RFM Analysis
df = pd.DataFrame(data)
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
last_date = df['transaction_date'].max() + pd.DateOffset(days=1)

from sklearn.cluster import KMeans
import pandas as pd

# Contoh data RFM
rfm_df = pd.DataFrame({
    'recency': [5, 10, 15],
    'frequency': [2, 3, 1],
    'monetary': [100, 200, 300]
})

# Data yang akan digunakan untuk clustering
X = rfm_df[['recency', 'frequency', 'monetary']]

# Gunakan 2 klaster, karena kita hanya memiliki 3 sampel
kmeans = KMeans(n_clusters=2, random_state=42)
rfm_df['cluster'] = kmeans.fit_predict(X)

# Tampilkan hasil clustering
print(rfm_df)


# Geospatial data for Map
geo_data = {
    'user_id': [1, 2, 3],
    'lat': [37.773972, 37.774159, 37.775000],
    'lon': [-122.431297, -122.429402, -122.433000]
}
df_geo = pd.DataFrame(geo_data)

# RFM Analysis
st.subheader("RFM Analysis")
st.dataframe(rfm_df)

# K-Means Clustering Result
st.subheader("Hasil K-Means Clustering")
st.write("Setiap pengguna dikelompokkan dalam cluster yang berbeda:")
st.dataframe(rfm_df)


