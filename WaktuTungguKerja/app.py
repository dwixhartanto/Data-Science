import streamlit as st
import joblib
import pandas as pd
import category_encoders as ce

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Kategori Masa Tunggu",
    page_icon="⏰",
    layout="wide" # Menggunakan layout "wide" agar ada lebih banyak ruang horizontal
)

st.subheader("⏰ Prediksi Kategori Masa Tunggu Pelamar")
st.markdown("""
Aplikasi ini memprediksi kategori masa tunggu pelamar berdasarkan data yang Anda masukkan.
""")

# --- 1. Muat Model AdaBoost dan Encoder (HARUS SAMA DENGAN SAAT TRAINING) ---

# Nama file model
MODEL_FILE = 'WaktuTungguKerja/adabostmodel.pkl'

@st.cache_resource # Gunakan st.cache_resource untuk memuat model hanya sekali
def load_model(model_path):
    try:
        model = joblib.load(model_path)
     #   st.success(f"Model '{model_path}' berhasil dimuat.")
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

adaboost_model = load_model(MODEL_FILE)

# Definisikan mapping untuk OrdinalEncoder yang digunakan pada FITUR KATEGORICAL (X)
feature_categorical_mapping = [
    {'col': 'Sosmed', 'mapping': {'tidak ada': 0, 'ada': 1}},
    {'col': 'biaya', 'mapping': {'pribadi': 0, 'beasiswa': 1}}, # Nama kolom 'biaya' sesuai klarifikasi terakhir
    {'col': 'b Inggris', 'mapping': {'sangat tinggi': 4, 'tinggi': 3, 'cukup tinggi': 2, 'rendah': 1, 'sangat rendah': 0}},
    {'col': 'IT', 'mapping': {'sangat tinggi': 4, 'tinggi': 3, 'cukup tinggi': 2, 'rendah': 1, 'sangat rendah': 0}},
    {'col': 'Komunikasi', 'mapping': {'sangat tinggi': 4, 'tinggi': 3, 'cukup tinggi': 2, 'rendah': 1, 'sangat rendah': 0}}
]

# Definisikan mapping terbalik untuk DEKODIFIKASI TARGET (y)
inverse_target_mapping = {
    3: '>18 bulan',
    2: '6-18 bulan',
    1: '<6 bulan'
}

# --- Ekstrak nama kolom kategorikal yang akan di-encode ---
categorical_cols_to_encode = [m['col'] for m in feature_categorical_mapping]

# Inisialisasi dan Fit encoder untuk fitur kategorikal (X)
@st.cache_resource
def get_encoder(mapping_data, cols_to_encode):
    encoder = ce.OrdinalEncoder(
        cols=cols_to_encode,
        return_df=True,
        mapping=mapping_data
    )
    dummy_data_dict = {}
    for m in mapping_data:
        col_name = m['col']
        first_value = list(m['mapping'].keys())[0]
        dummy_data_dict[col_name] = [first_value]
    
    dummy_df = pd.DataFrame(dummy_data_dict)
    encoder.fit(dummy_df)
    return encoder

encoder_X = get_encoder(feature_categorical_mapping, categorical_cols_to_encode)


# --- 2. Input dari Pengguna via Streamlit (Menggunakan Kolom) ---
st.header("Masukkan Data Pelamar:")

# Fungsi pembantu untuk mendapatkan opsi selectbox
def get_selectbox_options(col_name):
    for m in feature_categorical_mapping:
        if m['col'] == col_name:
            return list(m['mapping'].keys())
    return []

# Membuat 2 kolom untuk input
col1, col2 = st.columns(2) # Membagi layar menjadi 2 kolom dengan lebar yang sama

with col1:
    st.subheader("Informasi Utama")
    ipk = st.slider("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01,
                    help="Indeks Prestasi Kumulatif (0.0 - 4.0)")
    sosmed = st.selectbox("Tingkat Sosmed", options=get_selectbox_options('Sosmed'))
    biaya_sumber = st.selectbox("Sumber Biaya", options=get_selectbox_options('biaya'), key='sumber_biaya_input')

with col2:
    st.subheader("Kemampuan Tambahan")
    b_inggris = st.selectbox("Kemampuan Bahasa Inggris", options=get_selectbox_options('b Inggris'))
    it = st.selectbox("Kemampuan IT", options=get_selectbox_options('IT'))
    komunikasi = st.selectbox("Kemampuan Komunikasi", options=get_selectbox_options('Komunikasi'))

# Tombol Prediksi di luar kolom agar selalu di bawah input
if st.button("Prediksi Kategori Masa Tunggu"):
    # Buat DataFrame dari input user
    user_data_raw = pd.DataFrame({
        'IPK': [ipk],
        'Sosmed': [sosmed],
        'biaya': [biaya_sumber],
        'b Inggris': [b_inggris],
        'IT': [it],
        'Komunikasi': [komunikasi]
    })

    try:
        X_numeric = user_data_raw[['IPK']].copy()
        X_categorical_raw = user_data_raw[categorical_cols_to_encode].copy()
        
        X_categorical_encoded = encoder_X.transform(X_categorical_raw)
        
        final_X_for_prediction = pd.concat([X_numeric, X_categorical_encoded], axis=1)
        
        st.subheader("Hasil Prediksi:")
   #     st.dataframe(final_X_for_prediction)

        # Lakukan Prediksi
        prediksi_numerik = adaboost_model.predict(final_X_for_prediction)

        # Dekode Hasil Prediksi
        prediksi_label = inverse_target_mapping.get(prediksi_numerik[0], 'Kategori Tidak Dikenal')

        st.success(f"**Kategori masa tunggu yang diprediksi adalah: {prediksi_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.warning("Pastikan semua input sesuai dengan pilihan yang tersedia dan model sudah dilatih dengan struktur kolom yang sama.")

st.markdown("---")
st.caption("Aplikasi Prediksi Masa Tunggu - Dibuat dengan Streamlit")
