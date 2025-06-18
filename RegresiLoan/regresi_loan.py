import streamlit as st
import pandas as pd
import joblib

# --- Judul Aplikasi ---
st.title("Aplikasi Prediksi Income Pinjaman")

# --- Muat Model ---

model = joblib.load('RegresiLoan/regresi_loan.pkl')


# --- Input Pengguna ---
st.header("Masukkan Data untuk Prediksi")

col1, col2 = st.columns(2)

with col1:
    new_age = st.number_input("Masukkan nilai Age (Usia):", min_value=0, max_value=120, value=30)
with col2:
    new_experience = st.number_input("Masukkan nilai Experience (Pengalaman):", min_value=0, max_value=100, value=5)

# --- Tombol Prediksi ---
if st.button("Prediksi Income"):
    try:
        # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
        new_data_df = pd.DataFrame([[new_age, new_experience]], columns=['Age', 'Experience'])

        # Lakukan prediksi menggunakan model yang sudah dilatih
        predicted_income = model.predict(new_data_df)

        st.subheader("Hasil Prediksi:")
        st.write(f"Untuk Age = **{new_age}** dan Experience = **{new_experience}**:")
        # predicted_income adalah array 2D, ambil nilai tunggalnya
        st.success(f"Prediksi Income adalah: **${predicted_income[0][0]:,.2f}**")

    except ValueError:
        st.error("Input tidak valid. Harap masukkan angka.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- Informasi Tambahan (Opsional) ---
st.markdown("""
---
*Aplikasi ini memprediksi Income berdasarkan Age (Usia) dan Experience (Pengalaman) menggunakan model regresi yang telah dilatih.*
""")
