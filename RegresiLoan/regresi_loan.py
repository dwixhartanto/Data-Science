import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Konfigurasi Halaman (Opsional, untuk lebar) ---
st.set_page_config(layout="wide") # Mengatur layout halaman menjadi lebar

# --- Judul Aplikasi ---
st.title("Aplikasi Prediksi Income Pinjaman")

# --- Muat Model ---
try:
    # Sesuaikan path jika folder model Anda ada di sub-direktori lain
    model = joblib.load('RegresiLoan/regresi_loan.pkl')
    st.success("Model 'regresi_loan.pkl' berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: Model 'regresi_loan.pkl' tidak ditemukan. Pastikan file model ada di direktori yang benar.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Mengatur Layout dengan Kolom ---
# Buat dua kolom utama: satu untuk input, satu untuk output/grafik
input_column, output_column = st.columns([1, 2]) # Ratio 1:2, input 1/3 lebar, output 2/3 lebar

with input_column:
    st.header("Masukkan Data untuk Prediksi")

    new_age = st.number_input("Masukkan nilai Age (Usia):", min_value=0, max_value=120, value=30)
    new_experience = st.number_input("Masukkan nilai Experience (Pengalaman):", min_value=0, max_value=100, value=5)

    # Simpan hasil prediksi di session state agar bisa diakses setelah button ditekan
    # dan grafik bisa direfresh tanpa menekan button lagi
    if 'predicted_income' not in st.session_state:
        st.session_state.predicted_income = None
        st.session_state.new_age = None
        st.session_state.new_experience = None

    if st.button("Prediksi Income"):
        try:
            new_data_df = pd.DataFrame([[new_age, new_experience]], columns=['Age', 'Experience'])
            predicted_income_val = model.predict(new_data_df)

            # Simpan hasil ke session state
            st.session_state.predicted_income = predicted_income_val[0][0]
            st.session_state.new_age = new_age
            st.session_state.new_experience = new_experience

            st.success(f"Prediksi Income Dihitung!")
            # st.experimental_rerun() # Opsional: paksa rerun untuk update output column
            
        except ValueError:
            st.error("Input tidak valid. Harap masukkan angka.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

with output_column:
    st.header("Hasil Prediksi & Visualisasi")

    if st.session_state.predicted_income is not None:
        st.write(f"Untuk Age = **{st.session_state.new_age}** dan Experience = **{st.session_state.new_experience}**:")
        st.success(f"Prediksi Income adalah: **${st.session_state.predicted_income:,.2f}**")

        st.markdown("---")

        # --- Visualisasi Hasil Prediksi (Point plot) ---
        st.subheader("Visualisasi Prediksi Anda")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=[st.session_state.new_age], y=[st.session_state.predicted_income], s=200, color='red', label='Prediksi Anda', ax=ax)
        ax.set_title(f'Prediksi Income vs. Age (Experience: {st.session_state.new_experience})')
        ax.set_xlabel('Age')
        ax.set_ylabel('Predicted Income ($)')
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("---")

        # --- Visualisasi Rentang Prediksi (Contoh: Variasi Age dengan Experience tetap) ---
        st.subheader("Bagaimana Income Berubah Seiring Variasi Usia?")
        st.write(f"Dengan Experience tetap pada **{st.session_state.new_experience}**")

        age_range = np.linspace(0, 100, 100)
        range_data_df = pd.DataFrame({
            'Age': age_range,
            'Experience': st.session_state.new_experience
        })

        predicted_income_range = model.predict(range_data_df)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=age_range, y=predicted_income_range.flatten(), ax=ax2)
        sns.scatterplot(x=[st.session_state.new_age], y=[st.session_state.predicted_income], s=200, color='red', label='Prediksi Anda', ax=ax2, zorder=5)

        ax2.set_title(f'Perubahan Income vs. Age (Experience: {st.session_state.new_experience})')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Predicted Income ($)')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

# --- Informasi Tambahan (Opsional) ---
st.markdown("""
---
*Aplikasi ini memprediksi Income berdasarkan Age (Usia) dan Experience (Pengalaman) menggunakan model regresi yang telah dilatih.*
""")
