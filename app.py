import os
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_image
from feature_extraction import extract_all_features
from grading import grade_card
from utils import load_model

st.set_page_config(page_title="PokéScan | Card Authenticator", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    /* Mengatur padding atas agar tidak kepotong oleh top bar */
    .block-container {
        padding-top: 5.5rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* OVERRIDE WARNA VERSI DARK MODE (Hitam Harmonis & Kuning Stabilo) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #11131E !important; /* Hitam tidak terlalu pekat */
            color: #FAFAFA !important;
        }
        
        /* Mempercantik kartu metrik dengan aksen Kuning Stabilo */
        div[data-testid="metric-container"] {
            background-color: #1A1D2C !important;
            border: 1px solid #33354A !important;
            border-left: 5px solid #CCFF00 !important; /* Kuning Stabilo */
            padding: 12px 18px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        
        /* Mengubah warna font label metrik agar kontras */
        div[data-testid="stMetricLabel"] {
            color: #8A8D93 !important;
        }
        
        /* Desain tombol utama Kuning Stabilo */
        div.stButton > button:first-child {
            background-color: #CCFF00 !important; /* Kuning Stabilo */
            color: #0B0C13 !important; /* Teks gelap harmonis */
            font-weight: 800;
            border-radius: 8px;
            border: none;
            padding: 12px 24px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 12px rgba(204, 255, 0, 0.2);
            letter-spacing: 0.05em;
        }
        
        div.stButton > button:first-child:hover {
            background-color: #DDFF33 !important;
            transform: translateY(-2px);
            box-shadow: 0px 6px 18px rgba(204, 255, 0, 0.4);
        }
        
        /* Warna teks judul utama */
        .main-title {
            color: #FAFAFA;
        }
        
        /* Garis putus-putus box watermark panel kosong */
        .placeholder-box {
            border: 2px dashed #33354A !important;
            background-color: rgba(26, 29, 44, 0.4) !important;
        }
    }
    
    /* STYLE KELAS UMUM */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0px;
        letter-spacing: -0.03em;
    }
    .sub-title {
        color: #8A8D93;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = {0: "REAL", 1: "FAKE"}

@st.cache_resource
def init_model():
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    try:
        model, pipeline = load_model(MODEL_DIR)
        return model, pipeline, True
    except Exception as e:
        return None, None, False

model, pipeline, MODEL_LOADED = init_model()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/98/International_Pok%C3%A9mon_logo.svg", width=180)
    st.markdown("### ⚙️ Panel Kontrol")
    st.write("Pilih metode input gambar:")
    
    input_mode = st.radio("Sumber Gambar:", ("📂 Upload File", "📸 Webcam"), label_visibility="collapsed")
    st.markdown("---")
    
    if not MODEL_LOADED:
        st.error("🚨 Model (.pkl) tidak ditemukan!")
    else:
        st.success("✅ Sistem Autentikasi Siap")
        
    st.caption("© 2026 PokéScan by Kelompok 4")

st.markdown('<p class="main-title">⚡ PokéScan</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Pokémon Card Authenticator & Grader</p>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.warning("Menunggu model dimuat. Pastikan file model SVM berada di direktori `models/`.")
    st.stop()

img_file_buffers = []

if input_mode == "📂 Upload File":
    uploaded_files = st.file_uploader(
        "Seret & lepas hingga 10 gambar kartu Pokémon di sini", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("⚠️ Hanya 10 file pertama yang akan diproses oleh sistem.")
            img_file_buffers = uploaded_files[:10]
        else:
            img_file_buffers = uploaded_files
else:
    webcam_file = st.camera_input("Posisikan kartu secara simetris di depan kamera")
    if webcam_file is not None:
        img_file_buffers = [webcam_file]

if img_file_buffers:
    st.markdown("---")
    
    col_img, col_space, col_res = st.columns([1, 0.08, 1.2])

    with col_img:
        st.markdown(f"#### 🖼️ Pratinjau Gambar ({len(img_file_buffers)} File)")
        
        if len(img_file_buffers) > 1:
            gal_cols = st.columns(3)
            for idx, buf in enumerate(img_file_buffers):
                with gal_cols[idx % 3]:
                    img_preview = Image.open(buf)
                    st.image(img_preview, caption=buf.name, use_container_width=True)
        else:
            img_preview = Image.open(img_file_buffers[0])
            st.image(img_preview, caption=img_file_buffers[0].name, use_container_width=True)

    with col_res:
        st.markdown("#### 📊 Panel Hasil Analisis")
        
        scan_clicked = st.button("🔍 SCAN SEMUA KARTU SEKARANG", type="primary")
        
        if scan_clicked:
            tmp_paths = []
            
            with st.spinner(f'Sedang menganalisis {len(img_file_buffers)} gambar secara bergantian...'):
                for idx, file_buf in enumerate(img_file_buffers):
                    image = Image.open(file_buf).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    tmp_path = os.path.join(BASE_DIR, f"_tmp_capture_{idx}.jpg")
                    cv2.imwrite(tmp_path, img_bgr)
                    tmp_paths.append(tmp_path)

                    try:
                        img_color, img_gray = preprocess_image(tmp_path)
                        
                        with st.expander(f"📋 Hasil Analisis Kartu #{idx+1} — {file_buf.name}", expanded=True):
                            if img_color is None:
                                st.error("Gagal memproses gambar. Pastikan objek kartu terlihat seutuhnya.")
                                continue
                            
                            features = extract_all_features(img_color, img_gray)
                            scaler = pipeline["scaler"]
                            pca = pipeline.get("pca", None)
                            
                            X = features.reshape(1, -1)
                            X_scaled = scaler.transform(X)
                            if pca is not None:
                                X_scaled = pca.transform(X_scaled)

                            prediction = int(model.predict(X_scaled)[0])
                            proba = model.predict_proba(X_scaled)[0]
                            confidence = float(proba[prediction]) * 100
                            
                            grade_result = grade_card(img_bgr)
                            label = CLASS_NAMES[prediction]
                            
                            if label == "REAL":
                                st.markdown(
                                    f"""<div style="background:rgba(61, 255, 160, 0.08); border:1px solid #3dffa0; padding:12px; border-radius:8px; text-align:center; margin-bottom:15px;">
                                        <h3 style="color:#3dffa0; margin:0; font-size:1.3rem;">✅ KARTU ASLI (REAL)</h3>
                                        <p style="color:#E8EAF0; margin:0; font-size:0.85rem; opacity:0.85;">Confidence Level: {confidence:.1f}%</p>
                                    </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f"""<div style="background:rgba(255, 79, 109, 0.08); border:1px solid #ff4f6d; padding:12px; border-radius:8px; text-align:center; margin-bottom:15px;">
                                        <h3 style="color:#ff4f6d; margin:0; font-size:1.3rem;">🚨 KARTU PALSU (FAKE)</h3>
                                        <p style="color:#E8EAF0; margin:0; font-size:0.85rem; opacity:0.85;">Confidence Level: {confidence:.1f}%</p>
                                    </div>""", unsafe_allow_html=True)

                            t1, t2 = st.tabs(["🌟 Hasil Kondisi Fisik", "📈 Metrik Model AI"])
                            
                            with t1:
                                st.markdown(f"##### Kesimpulan Grade: **{grade_result['grade']}** `({grade_result['total_score']}/100)`")
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Centering", f"{grade_result['centering']['score']:.1f}")
                                m2.metric("Corners", f"{grade_result['corners']['score']:.1f}")
                                m3.metric("Edge Wear", f"{grade_result['edge_wear']['score']:.1f}")
                                
                            with t2:
                                st.caption("Distribusi probabilitas klasifikasi biner model:")
                                st.write(f"Persentase REAL: **{proba[0]*100:.1f}%**")
                                st.progress(float(proba[0]))
                                st.write(f"Persentase FAKE: **{proba[1]*100:.1f}%**")
                                st.progress(float(proba[1]))
                                
                    except Exception as e:
                        st.error(f"Terjadi kegagalan komputasi pada File #{idx+1}: {e}")
            
            for t_path in tmp_paths:
                if os.path.exists(t_path):
                    os.remove(t_path)
            st.toast("Seluruh gambar selesai dianalisis!", icon="🎉")
        else:
            st.markdown("""
                <div class="placeholder-box" style="text-align: center; padding: 50px 20px; margin-top: 20px; border: 2px dashed #252a42; border-radius: 12px; background-color: rgba(20, 23, 38, 0.5);">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg" 
                        width="110" 
                        style="opacity: 0.35; margin-bottom: 15px;"/>
                    <h5 style="color: #5c6388; font-weight: 600; margin-bottom: 5px;">Sistem PokéScan Siap</h5>
                    <p style="color: #5c6388; font-size: 0.85rem; max-width: 320px; margin: 0 auto;">
                        Silakan tekan tombol di atas untuk memulai kalkulasi ekstraksi fitur dan grading fisik kartu secara bersamaan.
                    </p>
                </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 70px 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg" 
                width="130" 
                style="opacity: 0.15; margin-bottom: 20px;"/>
            <h4 style="color: #5c6388; font-weight: 500;">Menunggu Input Gambar Kartu</h4>
            <p style="color: #43496a; font-size: 0.9rem;">
                Silakan gunakan menu di <b>Sidebar kiri</b> untuk memilih mengunggah file gambar (maksimal 10) atau mengaktifkan kamera device Anda.
            </p>
        </div>
    """, unsafe_allow_html=True)