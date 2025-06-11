import streamlit as st
import pickle
import numpy as np
import os # Import modul 'os' untuk operasi sistem file

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Waktu Tunggu Kerja",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title('Aplikasi Prediksi Waktu Tunggu Kerja')

# Fungsi untuk memuat model menggunakan cache Streamlit
# Ini memastikan model hanya dimuat sekali saat aplikasi dijalankan,
# bukan setiap kali ada interaksi pengguna, sehingga aplikasi lebih ringan.
@st.cache_resource
def load_model():
    """Memuat model KNN dari file pickle."""
    model_path = 'model_knn.pkl'
    if not os.path.exists(model_path):
        st.error(f"Error: '{model_path}' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
        st.stop() # Hentikan eksekusi aplikasi jika model tidak ditemukan
    try:
        with open(model_path, 'rb') as file:
            knn_tuned = pickle.load(file)
        st.success("Model 'model_knn.pkl' berhasil dimuat!")
        return knn_tuned
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop() # Hentikan eksekusi aplikasi jika gagal memuat model

# Memuat model
knn_tuned = load_model()

# Debugging: Cetak daftar file di direktori kerja
st.subheader("Informasi Debugging:")
st.write(f"Direktori kerja saat ini: `{os.getcwd()}`")
st.write("Daftar file di direktori saat ini:")
try:
    st.code(os.listdir('.')) # Menampilkan daftar file dalam format kode
except Exception as e:
    st.warning(f"Tidak dapat membaca daftar direktori: {e}")
st.markdown("---") # Garis pemisah untuk debugging info

st.write('Aplikasi ini memprediksi waktu tunggu kerja alumni berdasarkan fakultas, IPK, dan lama studi.')

# Input fields untuk pengguna
fakultas = st.selectbox(
    'Pilih Fakultas:',
    ('Ekonomi', 'Hukum', 'Ilmu Kesehatan', 'Keguruan dan Ilmu Pendidikan', 'Perikanan', 'Pertanian', 'Lainnya')
)

ipk = st.number_input(
    'Masukkan IPK (contoh: 3.50)',
    min_value=0.00,
    max_value=4.00,
    value=3.00,
    step=0.01,
    format="%.2f" # Format untuk dua angka di belakang koma
)

# Menggunakan kolom untuk input lama studi agar lebih rapi
col1, col2 = st.columns(2)
with col1:
    lama_studi_thn = st.number_input('Lama Studi (Tahun):', min_value=0, max_value=10, value=4, step=1)
with col2:
    lama_studi_bln = st.number_input('Lama Studi (Bulan):', min_value=0, max_value=11, value=0, step=1)

# Tombol untuk melakukan prediksi
if st.button('Prediksi Waktu Tunggu', help="Klik untuk mendapatkan prediksi waktu tunggu kerja Anda."):
    # Menyiapkan data input untuk model
    data_input = []

    # Mapping Fakultas ke nilai numerik
    if fakultas == 'Ekonomi':
        data_input.append(0)
    elif fakultas == 'Hukum':
        data_input.append(1)
    elif fakultas == 'Ilmu Kesehatan':
        data_input.append(2)
    elif fakultas == 'Keguruan dan Ilmu Pendidikan':
        data_input.append(3)
    elif fakultas == 'Perikanan':
        data_input.append(4)
    elif fakultas == 'Pertanian':
        data_input.append(5)
    else: # Untuk fakultas 'Lainnya'
        data_input.append(6)

    data_input.append(ipk)
    lama_studi = lama_studi_thn * 12 + lama_studi_bln # Konversi lama studi ke bulan
    data_input.append(lama_studi)

    # Mengubah data input menjadi array NumPy dan reshape untuk prediksi model
    data_input_np = np.array(data_input).reshape(1, -1)

    # Melakukan prediksi menggunakan model
    try:
        prediction = knn_tuned.predict(data_input_np)
        output = prediction[0]

        output2 = ""
        # Menentukan hasil prediksi berdasarkan output model
        if output == 0:
            output2 = "Waktu Tunggu Kerja Kurang Dari 6 Bulan"
        elif output == 1:
            output2 = "Waktu Tunggu Kerja Lebih Dari 6 Bulan dan Kurang Dari 18 Bulan"
        elif output == 2:
            output2 = "Waktu Tunggu Kerja Lebih Dari 18 Bulan"
        else:
            output2 = "Error: Prediksi tidak valid atau di luar rentang yang diketahui."

        # Menampilkan hasil prediksi kepada pengguna
        st.success(f"**Hasil Prediksi Alumni Fakultas {fakultas} dengan IPK {ipk:.2f} dan Masa Studi {lama_studi_thn} Tahun {lama_studi_bln} Bulan:**")
        st.info(f"**{output2}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}. Pastikan input data valid.")

st.markdown("---")
st.markdown("Aplikasi ini dibuat untuk tujuan demonstrasi dan mungkin tidak mencerminkan akurasi 100%.")
