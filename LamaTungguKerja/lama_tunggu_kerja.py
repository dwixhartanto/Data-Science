import streamlit as st
import pickle
import numpy as np

# Tambahkan ini

st.title('Aplikasi Prediksi Waktu Tunggu Kerja')

# Debugging: Cetak daftar file di direktori kerja
st.write("Files in current directory:")
st.write(os.listdir('.')) # Ini akan mencetak daftar file dan folder di direktori root aplikasi

# Load the trained KNN model
try:
    with open('model_knn.pkl', 'rb') as file:
        knn_tuned = pickle.load(file)
    st.success("Model 'model_knn.pkl' berhasil dimuat!") # Tambahkan ini jika berhasil
except FileNotFoundError:
    st.error("Error: 'model_knn.pkl' not found. Please make sure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# ... (sisa kode aplikasi Anda) ...

# Load the trained KNN model
#try:
 #   with open('model_knn.pkl', 'rb') as file:
  #      knn_tuned = pickle.load(file)
#except FileNotFoundError:
 #   st.error("Error: 'model_knn.pkl' not found. Please make sure the model file is in the same directory.")
  #  st.stop()

#st.title('Aplikasi Prediksi Waktu Tunggu Kerja')

st.write('Aplikasi ini memprediksi waktu tunggu kerja alumni berdasarkan fakultas, IPK, dan lama studi.')

# Input fields for the user
fakultas = st.selectbox(
    'Pilih Fakultas:',
    ('Ekonomi', 'Hukum', 'Ilmu Kesehatan', 'Keguruan dan Ilmu Pendidikan', 'Perikanan', 'Pertanian', 'Lainnya')
)

ipk = st.number_input('Masukkan IPK (contoh: 3.50)', min_value=0.00, max_value=4.00, value=3.00, step=0.01)

col1, col2 = st.columns(2)
with col1:
    lama_studi_thn = st.number_input('Lama Studi (Tahun):', min_value=0, max_value=10, value=4, step=1)
with col2:
    lama_studi_bln = st.number_input('Lama Studi (Bulan):', min_value=0, max_value=11, value=0, step=1)

if st.button('Prediksi Waktu Tunggu'):
    # Prepare the input data for the model
    data_input = []

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
    else:
        data_input.append(6)

    data_input.append(ipk)
    lama_studi = lama_studi_thn * 12 + lama_studi_bln
    data_input.append(lama_studi)

    # Convert to numpy array and reshape for prediction
    data_input_np = np.array(data_input).reshape(1, -1)

    # Make prediction
    try:
        prediction = knn_tuned.predict(data_input_np)
        output = prediction[0]

        output2 = ""
        if output == 0:
            output2 = "Waktu Tunggu Kerja Kurang Dari 6 Bulan"
        elif output == 1:
            output2 = "Waktu Tunggu Kerja Lebih Dari 6 Bulan dan Kurang Dari 18 Bulan"
        elif output == 2:
            output2 = "Waktu Tunggu Kerja Lebih Dari 18 Bulan"
        else:
            output2 = "Error: Prediksi tidak valid"

        st.success(f"**Hasil Prediksi Alumni Fakultas {fakultas} Dengan IPK {ipk:.2f} Dan Masa Studi {lama_studi_thn} Tahun {lama_studi_bln} Bulan:**")
        st.info(f"**{output2}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
