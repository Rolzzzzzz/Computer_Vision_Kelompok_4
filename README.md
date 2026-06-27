================================================================================
1. FILE DAN FOLDER YANG PERLU DIHAPUS (CLEANUP)
================================================================================
Hapus file dan folder lama berikut agar tidak bentrok dengan arsitektur Streamlit:
- Hapus file: index.html
- Hapus folder: templates/ (jika ada)
- Hapus folder: static/ (jika ada)

================================================================================
2. ISI FILE DEPENDENSI BARU (requirements.txt)
================================================================================
Pastikan file requirements.txt kamu berisi library berikut (tambahkan streamlit):

numpy
opencv-python
scikit-learn
Pillow
streamlit

================================================================================
3. ISI FILE UTAMA APLIKASI BARU (app.py)
================================================================================
Ganti seluruh isi file app.py kamu dengan kode Streamlit di bawah ini:

import os
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Setup direktori agar bisa import dari modul lain
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_image
from feature_extraction import extract_all_features
from grading import grade_card
from utils import load_model

# Configurasi Halaman Streamlit
st.set_page_config(page_title="PokéScan", page_icon="⚡", layout="centered")

CLASS_NAMES = {0: "REAL", 1: "FAKE"}

# Menggunakan cache agar model tidak perlu di-load berulang kali setiap ada interaksi
@st.cache_resource
def init_model():
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    try:
        model, pipeline = load_model(MODEL_DIR)
        return model, pipeline, True
    except Exception as e:
        return None, None, False

model, pipeline, MODEL_LOADED = init_model()

# Header Aplikasi
st.title("⚡ PokéScan — Card Authenticator")

if not MODEL_LOADED:
    st.error("Model belum dimuat. Pastikan file .pkl ada di folder models/ atau jalankan train.py dulu!")
    st.stop()
else:
    st.success("✅ Model berhasil dimuat!")

st.markdown("---")

# Memilih Mode Input
input_mode = st.radio("Pilih Mode Gambar:", ("📂 Upload File", "📸 Kamera (Webcam)"))

img_file_buffer = None
if input_mode == "📂 Upload File":
    img_file_buffer = st.file_uploader("Upload gambar kartu Pokémon Anda", type=["jpg", "jpeg", "png"])
else:
    img_file_buffer = st.camera_input("Ambil foto kartu langsung dari Webcam")

# Proses jika gambar sudah dimasukkan
if img_file_buffer is not None:
    # Membaca gambar menggunakan PIL lalu konversi ke OpenCV format (BGR)
    image = Image.open(img_file_buffer).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if input_mode == "📂 Upload File":
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    if st.button("🔍 Scan Kartu Sekarang", type="primary"):
        with st.spinner('Menganalisis kartu...'):
            # Simpan file sementara (karena preprocess_image membaca path file)
            tmp_path = os.path.join(BASE_DIR, "_tmp_capture.jpg")
            cv2.imwrite(tmp_path, img_bgr)

            try:
                # 1. Preprocessing
                img_color, img_gray = preprocess_image(tmp_path)
                if img_color is None:
                    st.error("Gagal memproses gambar. Pastikan pencahayaan bagus dan seluruh kartu terlihat.")
                else:
                    # 2. Feature Extraction
                    features = extract_all_features(img_color, img_gray)
                    
                    # 3. Predict Authenticity
                    scaler = pipeline["scaler"]
                    pca = pipeline.get("pca", None) # Ambil PCA jika digunakan
                    
                    X = features.reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    if pca is not None:
                        X_scaled = pca.transform(X_scaled)

                    prediction = int(model.predict(X_scaled)[0])
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = float(proba[prediction]) * 100
                    
                    # 4. Grading Kondisi Kartu
                    grade_result = grade_card(img_bgr)

                    # ==== TAMPILKAN HASIL UI ====
                    st.markdown("---")
                    st.subheader("🕵️ Hasil Autentikasi")
                    
                    col1, col2 = st.columns(2)
                    label = CLASS_NAMES[prediction]
                    
                    if label == "REAL":
                        col1.success(f"**Kartu Asli (REAL)**")
                    else:
                        col1.error(f"**Kartu Palsu (FAKE)**")
                        
                    col2.metric("Confidence Level", f"{confidence:.1f}%")

                    st.markdown("**Probabilitas Model:**")
                    st.write(f"REAL: {proba[0]*100:.1f}%")
                    st.progress(float(proba[0]))
                    st.write(f"FAKE: {proba[1]*100:.1f}%")
                    st.progress(float(proba[1]))

                    st.markdown("---")
                    st.subheader("✨ Kondisi Kartu (Grading)")
                    st.markdown(f"**Kesimpulan Grade:** `{grade_result['grade']}` (Skor Total: **{grade_result['total_score']} / 100**)")
                    
                    g_col1, g_col2, g_col3 = st.columns(3)
                    g_col1.metric("Centering", f"{grade_result['centering']['score']:.1f}")
                    g_col2.metric("Corners", f"{grade_result['corners']['score']:.1f}")
                    g_col3.metric("Edge Wear", f"{grade_result['edge_wear']['score']:.1f}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
            finally:
                # Bersihkan file temporary
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


================================================================================
4. ISI DOCUMENTASI BARU (README.md)
================================================================================
Ganti seluruh isi file README.md kamu dengan teks di bawah ini:

# ⚡ PokéScan — Pokémon Card Authenticator & Grader

PokéScan adalah aplikasi berbasis *Computer Vision* dan *Machine Learning* yang dirancang untuk menganalisis kartu Pokémon. Aplikasi ini dapat membedakan antara kartu asli dan palsu (autentikasi) sekaligus mengevaluasi kondisi fisik kartu tersebut (*grading*) secara otomatis.

Aplikasi ini sekarang dibangun menggunakan **Streamlit** untuk antarmuka pengguna yang interaktif, mendukung penggunaan *webcam* secara langsung maupun unggah gambar.

## ✨ Fitur Utama
* 🔍 **Autentikasi Kartu (Real vs Fake):** Menggunakan model Support Vector Machine (SVM) yang dilatih dengan ekstraksi fitur HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns), dan Color Histograms.
* 📏 **Sistem Grading Otomatis:** Mengevaluasi kondisi kartu berdasarkan 3 metrik utama:
  * **Centering:** Mengukur simetri batas (*border*) vertikal dan horizontal.
  * **Corners:** Mendeteksi keausan atau kerusakan pada sudut kartu.
  * **Edge Wear:** Memeriksa keutuhan tepi kartu.
* 📸 **Input Fleksibel:** Mendukung pengambilan gambar langsung dari **Webcam** atau **Upload File** gambar dari direktori lokal.

## 📂 Struktur Proyek Terkini
```text
.
├── dataset/                  # Folder berisi gambar latih & uji, serta label (CSV)
├── models/                   # Folder tempat penyimpanan model SVM (.pkl)
├── results/                  # Visualisasi metrik evaluasi (Confusion Matrix, dll)
├── app.py                    # Script utama aplikasi Streamlit (Web UI)
├── feature_extraction.py     # Logika ekstraksi fitur gambar (HOG, LBP, warna, ORB)
├── grading.py                # Algoritma pengukuran kondisi fisik kartu
├── predict.py                # Script CLI murni untuk prediksi (tanpa UI)
├── preprocessing.py          # Modul pembersihan dan pemotongan gambar (cropping)
├── train.py                  # Script untuk melatih ulang model SVM
├── utils.py                  # Fungsi bantuan (load model, file handling)
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Dokumentasi proyek