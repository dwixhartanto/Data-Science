import streamlit as st
import pandas as pd
import joblib

# --- Judul Aplikasi ---
st.title('Prediksi Kategori Waktu Lulus Mahasiswa')
st.write('Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus "Tepat Waktu" atau "Terlambat" berdasarkan beberapa parameter.')

# --- Memuat Model ---
try:
    with open('StudentPerformance/model_graduation.pkl', 'rb') as file:
        model = joblib.load(file)
    st.success('Model berhasil dimuat!')
except FileNotFoundError:
    st.error('File model_graduation.pkl tidak ditemukan. Pastikan file model berada di direktori yang sama.')
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f'Terjadi kesalahan saat memuat model: {e}')
    st.stop()

# --- Input Pengguna ---
st.header('Masukkan Data Mahasiswa:')

# Menggunakan st.number_input untuk input numerik
new_ACT = st.number_input('Nilai ACT composite score:', min_value=0.0, max_value=36.0, value=25.0)
new_SAT = st.number_input('Nilai SAT total score:', min_value=0.0, max_value=1600.0, value=1200.0)
new_GPA = st.number_input('Nilai rata-rata SMA (GPA):', min_value=0.0, max_value=4.0, value=3.5)
new_income = st.number_input('Pendapatan orang tua (USD):', min_value=0.0, value=50000.0)
new_education = st.number_input('Tingkat pendidikan orang tua (numerik, misal: 1=SD, 2=SMP, dst.):', min_value=1.0, value=3.0)

# --- Tombol Prediksi ---
if st.button('Prediksi Kategori Lulus'):
    # Buat DataFrame dari input baru
    new_data_df = pd.DataFrame(
        [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
        columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
    )

    # Lakukan prediksi
    try:
        predicted_code = model.predict(new_data_df)[0]
        label_mapping = {1: 'On Time (Tepat Waktu)', 0: 'Late (Terlambat)'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success('Prediksi berhasil!')
        st.write(f'**Prediksi kategori masa studi adalah: {predicted_label}**')
    except Exception as e:
        st.error(f'Terjadi kesalahan saat melakukan prediksi: {e}')

# --- Informasi Tambahan (Opsional) ---
st.markdown(
    """
    ---
    *Catatan: Model ini adalah contoh dan mungkin memerlukan penyesuaian lebih lanjut untuk akurasi yang lebih baik.*
    """
)
