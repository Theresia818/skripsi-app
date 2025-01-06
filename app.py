import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit_antd_components as sac
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(
    page_title="Prediksi Harga Saham",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Membaca dataset
datas = pd.read_excel('Dataset.xlsx')

# Normalisasi data
X = datas[['Tertinggi', 'Terendah']]
y = datas['Penutupan']
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Sidebar menu
with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('Home', icon="house"),
        sac.MenuItem('Datasset', icon='database-add', children=[
            sac.MenuItem('Dataset', icon='database'),
            sac.MenuItem('Data Normalisasi', icon='activity'),
            sac.MenuItem('Pola Inputan dan Target', icon='bi bi-database-fill-down'),
        ]),
        sac.MenuItem('Prediksi', icon="bi bi-bar-chart-line-fill")
    ], open_all=False)

# Halaman Home
if selected == "Home":
    col1, col2, col3 = st.columns([1, 10, 1])
    with col3:
        st.image('Logo.png', width=100)
    st.title("PREDIKSI HARGA PENUTUPAN SAHAM BANK BCA")
    st.write("Sistem prediksi Harga penutupan Saham Bank BCA menggunakan dataset yang berasal dari website penyedia sarana perdagangan saham indonesia yaitu PT Bursa Efek Indonesia yang dapat diakses melalui link berikut: https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham .")
    st.write("Dataset yang digunakan merupakan data harian harga saham pada Bank BCA yang berjumlah 1.185 data dimulai dari tanggal 02 Januari 2020 hingga 19 November 2024. Dataset tersebut memiliki 3 atribut yaitu harga tertinggi, harga terendah dan harga penutupan. Harga tertinggi dan harga terendah menjadi pola inputan sedangkan harga penutupan menjadi target untuk dilakukan prediksi.")
    st.subheader("Single Layer Percpetron")
    st.write("Single layer perceptron adalah model Artificial Neural Network yang melalui pembelajaran pengawasan pada sistem jaringan saraf. Model Single Layer Percpetron  memiliki satu lapisan input (input layer) dan satu lapisan output (output layer) tanpa lapisan tersembunyi (hidden layer)")

# Halaman Dataset
if selected == "Dataset":
    st.title("Dataset")
    st.write("Berikut adalah dataset atau data aktual yang digunakan untuk prediksi harga penutupan saham Bank BCA. ")
    st.subheader("Grafik Dataset")
    st.line_chart(datas)
    st.subheader("Tabel Dataset")
    st.dataframe(datas.rename(columns={'Tertinggi': 'Harga Tertinggi', 'Terendah': 'Harga Terendah', 'Penutupan': 'Harga Penutupan'}), width=700, hide_index=True)

# Halaman Data Normalisasi
if selected == "Data Normalisasi":
    st.title("Data Normalisasi")
    st.write("Data Normalisasi adalah proses mengubah skala data aktual atau asli ke rentang nilai antara 0 dan 1. Berikut adalah hasil Normalisasi Data : ")
    scaled_datas = pd.DataFrame(X_scaled, columns=['Harga Tertinggi', 'Harga Terendah'])
    scaled_datas['Harga Penutupan'] = y_scaled
    st.dataframe(scaled_datas, width=700, hide_index=True)
    st.line_chart(scaled_datas)

# Halaman Pola Inputan dan Target
if selected == "Pola Inputan dan Target":
    st.title("Pola Inputan dan Target")
    st.write("Harga tertinggi sebagai inputan X1, harga terendah sebagai inputan X2 dan harga penutupan sebagai Y atau target.")
    st.write("Proses prediksi melibatkan data X1 dan X2 yang sudah dinormalisasi sebagai input. Untuk mengembalikan angka hasil prediksi menjadi nilai asli, dilakukan proses denormalisasi")
    scaled_datas = pd.DataFrame(X_scaled, columns=['X1', 'X2'])
    scaled_datas['Y'] = y_scaled
    st.dataframe(scaled_datas, width=700, hide_index=True)

# Halaman Prediksi
if selected == "Prediksi":
    st.title("Prediksi")
    tertinggi = st.number_input("Saham Tertinggi", min_value=1, max_value=1000000, step=1)
    terendah = st.number_input("Saham Terendah", min_value=1, max_value=1000000, step=1)
    if st.button("Prediksi"):
        data_input = np.array([[tertinggi, terendah]])
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

        # Cek apakah file model ada
        model_path = 'Model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            hasil = model.predict(data_input)
            st.subheader(f"Prediksi Penutupan Saham: {int(hasil[0][0])}")
        else:
            st.error(f"Model file '{model_path}' not found. Please ensure the model file exists at the specified path.")
