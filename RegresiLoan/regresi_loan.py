import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Untuk membuat rentang data

# --- Judul Aplikasi ---
st.title("Aplikasi Prediksi Income Pinjaman")

# --- Muat Model ---
try:
    model = joblib.load('RegresiLoan/regresi_loan.pkl') # Perbaikan path sesuai diskusi sebelumnya
    st.success("Model 'regresi_loan.pkl' berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: Model 'regresi_loan.pkl' tidak ditemukan. Pastikan file model ada di direktori yang benar.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

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
        st.success(f"Prediksi Income adalah: **${predicted_income[0][0]:,.2f}**")

        # --- Visualisasi Hasil Prediksi ---
        st.subheader("Visualisasi Prediksi Anda")

        # Visualisasi sederhana (Point plot)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=[new_age], y=[predicted_income[0][0]], s=200, color='red', label='Prediksi Anda', ax=ax)
        ax.set_title(f'Prediksi Income vs. Age (Experience: {new_experience})')
        ax.set_xlabel('Age')
        ax.set_ylabel('Predicted Income ($)')
        ax.grid(True)
        st.pyplot(fig) # Tampilkan plot di Streamlit

        st.markdown("---")

        # --- Visualisasi Rentang Prediksi (Contoh: Variasi Age dengan Experience tetap) ---
        st.subheader("Bagaimana Income Berubah Seiring Variasi Usia?")
        st.write(f"Dengan Experience tetap pada **{new_experience}**")

        # Buat rentang data Age
        age_range = np.linspace(0, 100, 100) # 100 titik data dari 0 sampai 100 tahun
        # Buat DataFrame untuk prediksi rentang
        range_data_df = pd.DataFrame({
            'Age': age_range,
            'Experience': new_experience # Experience tetap
        })

        # Prediksi untuk rentang data
        predicted_income_range = model.predict(range_data_df)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=age_range, y=predicted_income_range.flatten(), ax=ax2) # Gunakan flatten() karena predicted_income_range bisa 2D
        # Tandai posisi prediksi pengguna
        sns.scatterplot(x=[new_age], y=[predicted_income[0][0]], s=200, color='red', label='Prediksi Anda', ax=ax2, zorder=5)

        ax2.set_title(f'Perubahan Income vs. Age (Experience: {new_experience})')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Predicted Income ($)')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    except ValueError:
        st.error("Input tidak valid. Harap masukkan angka.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- Informasi Tambahan (Opsional) ---
st.markdown("""
---
*Aplikasi ini memprediksi Income berdasarkan Age (Usia) dan Experience (Pengalaman) menggunakan model regresi yang telah dilatih.*
""")
