import streamlit as st
import pandas as pd
import numpy as np
import joblib # Mengimpor joblib untuk memuat model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Income",
    page_icon="💰",
    layout="centered"
)

# --- Judul Aplikasi ---
st.title("💰 Prediksi Income (Pendapatan)")
st.markdown("Aplikasi ini memprediksi *income* berdasarkan *Age* (Usia) dan *Experience* (Pengalaman).")

# --- Memuat Model yang Sudah Dilatih ---
# CATATAN PENTING:
# Pastikan file model 'regresi_loan.pkl' ada di repositori GitHub Anda
# di lokasi yang sama dengan file app.py ini.
model_path = 'Data-Science/regresi_loan.pkl'

try:
    model = joblib.load(model_path)
    st.success(f"Model '{model_path}' berhasil dimuat!")
except FileNotFoundError:
    st.error(f"Error: File model '{model_path}' tidak ditemukan.")
    st.warning("Pastikan file `regresi_loan.pkl` ada di repositori GitHub Anda di lokasi yang sama dengan aplikasi ini.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model valid dan kode pemuatan sudah benar. Error: {e}")
    st.stop() # Hentikan eksekusi jika ada error lain saat memuat model

# --- Input Pengguna ---
st.header("Masukkan Data Baru")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (Usia)",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Masukkan usia Anda."
    )

with col2:
    experience = st.number_input(
        "Experience (Pengalaman) dalam Tahun",
        min_value=0,
        max_value=50,
        value=5,
        step=0.5,
        help="Masukkan total tahun pengalaman kerja Anda."
    )

# --- Tombol Prediksi ---
if st.button("Prediksi Income"):
    # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
    # Pastikan nama kolom 'Age' dan 'Experience' sesuai dengan yang digunakan saat melatih model
    new_data_df = pd.DataFrame([[age, experience]], columns=['Age', 'Experience'])

    try:
        # Lakukan prediksi menggunakan model yang sudah dilatih
        predicted_income = model.predict(new_data_df)

        # Asumsi output model adalah array 2D, ambil nilai tunggalnya
        # Format angka menjadi mata uang dengan dua desimal
        formatted_income = f"${predicted_income[0][0]:,.2f}"

        st.subheader("Hasil Prediksi")
        st.success(f"Untuk Age = **{int(age)}** dan Experience = **{experience}** tahun,")
        st.success(f"Prediksi Income adalah: **{formatted_income}**")
        st.balloons() # Efek balon saat prediksi berhasil

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.warning("Pastikan input valid dan model berfungsi dengan benar untuk data ini.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit.")
