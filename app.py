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
    .block-container {
        padding-top: 5.5rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 12px 18px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    }
    .main-title { font-size: 2.8rem; font-weight: 900; margin-bottom: 0px; letter-spacing: -0.03em; }
    .sub-title  { font-size: 1.05rem; margin-bottom: 25px; opacity: 0.7; }
    .section-header {
        font-size: 1rem; font-weight: 700; margin: 12px 0 8px 0;
        padding: 6px 10px;
        border-radius: 6px;
        background: rgba(128,128,128,0.08);
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
    except Exception:
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
        st.caption("Jalankan `train.py` terlebih dahulu.")
    else:
        st.success("✅ Sistem Autentikasi SVM Siap")

    st.caption("© 2026 PokéScan by Kelompok 4")

st.markdown('<p class="main-title">⚡ PokéScan</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Pokémon Card Authenticator & Grader</p>', unsafe_allow_html=True)

img_file_buffers = []

if input_mode == "📂 Upload File":
    uploaded_files = st.file_uploader(
        "Seret & lepas hingga 10 gambar kartu Pokémon di sini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("⚠️ Hanya 10 file pertama yang akan diproses.")
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
                    st.image(Image.open(buf), caption=buf.name, use_container_width=True)
        else:
            st.image(Image.open(img_file_buffers[0]), caption=img_file_buffers[0].name, use_container_width=True)

    with col_res:
        st.markdown("#### 📊 Panel Hasil Analisis")
        scan_clicked = st.button("🔍 SCAN SEMUA KARTU SEKARANG", type="primary")

        if scan_clicked:
            tmp_paths = []
            with st.spinner(f'Menganalisis {len(img_file_buffers)} gambar...'):
                for idx, file_buf in enumerate(img_file_buffers):
                    image   = Image.open(file_buf).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    tmp_path = os.path.join(BASE_DIR, f"_tmp_capture_{idx}.jpg")
                    cv2.imwrite(tmp_path, img_bgr)
                    tmp_paths.append(tmp_path)

                    try:
                        img_color, img_gray = preprocess_image(tmp_path)

                        with st.expander(f"📋 Hasil Kartu #{idx+1} — {file_buf.name}", expanded=True):
                            if img_color is None:
                                st.error("Gagal memproses gambar.")
                                continue

                            features = extract_all_features(img_color, img_gray)
                            scaler   = pipeline["scaler"]
                            pca      = pipeline.get("pca")
                            X        = features.reshape(1, -1)
                            X_scaled = scaler.transform(X)
                            if pca is not None:
                                X_scaled = pca.transform(X_scaled)
                                
                            prediction = int(model.predict(X_scaled)[0])
                            proba      = model.predict_proba(X_scaled)[0]
                            confidence = float(proba[prediction]) * 100
                            
                            proba_real = float(proba[0])
                            proba_fake = float(proba[1])
                            method_note = "SVM Model"

                            label = CLASS_NAMES[prediction]
                            grade_result = grade_card(img_bgr)

                            st.markdown(f"<p class='section-header'>🔐 Autentikasi Kartu <span style='font-weight:400;font-size:0.8rem;opacity:0.6;'>({method_note})</span></p>", unsafe_allow_html=True)

                            if label == "REAL":
                                st.markdown(
                                    f"""<div style="background:rgba(46,204,113,0.15);border:2px solid #2ecc71;padding:12px;border-radius:8px;text-align:center;margin-bottom:10px;">
                                        <h3 style="color:#27ae60;margin:0;font-size:1.3rem;">✅ KARTU ASLI (REAL)</h3>
                                        <p style="margin:4px 0 0 0;font-size:0.85rem;opacity:0.85;">Confidence: {confidence:.1f}%</p>
                                    </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f"""<div style="background:rgba(231,76,60,0.15);border:2px solid #e74c3c;padding:12px;border-radius:8px;text-align:center;margin-bottom:10px;">
                                        <h3 style="color:#c0392b;margin:0;font-size:1.3rem;">🚨 KARTU PALSU (FAKE)</h3>
                                        <p style="margin:4px 0 0 0;font-size:0.85rem;opacity:0.85;">Confidence: {confidence:.1f}%</p>
                                    </div>""", unsafe_allow_html=True)

                            with st.expander("📈 Detail Probabilitas / Skor", expanded=False):
                                st.write(f"P(REAL): **{proba_real*100:.1f}%**")
                                st.progress(proba_real)
                                st.write(f"P(FAKE): **{proba_fake*100:.1f}%**")
                                st.progress(proba_fake)
                                st.markdown("---")
                                st.caption("🔬 Ekstraksi Fitur Global (HOG + LBP + Color Histogram + ORB)")
                                st.caption(f"Total dimensi vektor input setelah normalisasi: **{features.shape[0]}**")

                            st.markdown("---")
                            st.markdown("<p class='section-header'>🌟 Grading Kondisi Fisik Kartu</p>", unsafe_allow_html=True)

                            grade_label = grade_result["grade"]
                            grade_colors = {
                                "Mint":      ("#1abc9c", "rgba(26,188,156,0.15)"),
                                "Near Mint": ("#3498db", "rgba(52,152,219,0.15)"),
                                "Good":      ("#f39c12", "rgba(243,156,18,0.15)"),
                                "Poor":      ("#e74c3c", "rgba(231,76,60,0.15)"),
                            }
                            g_color, g_bg = grade_colors.get(grade_label, ("#888", "rgba(128,128,128,0.1)"))
                            st.markdown(
                                f"""<div style="background:{g_bg};border:2px solid {g_color};padding:10px 16px;border-radius:8px;text-align:center;margin-bottom:12px;">
                                    <h3 style="color:{g_color};margin:0;font-size:1.2rem;">🏅 Grade: {grade_label}</h3>
                                    <p style="margin:4px 0 0 0;font-size:0.85rem;opacity:0.85;">Total Score: {grade_result['total_score']}/100</p>
                                </div>""", unsafe_allow_html=True)

                            m1, m2, m3 = st.columns(3)
                            centering_score = grade_result["centering"]["score"]
                            centering_note  = grade_result["centering"].get("note", "ok")
                            c_delta = None
                            if centering_note == "card_fills_frame":   c_delta = "Frame penuh (estimasi)"
                            elif centering_note == "no_card_detected": c_delta = "Kartu tidak terdeteksi"

                            m1.metric("Centering (40%)", f"{centering_score:.1f}",
                                    delta=c_delta, delta_color="off" if c_delta else "normal",
                                    help="Simetri border kiri-kanan dan atas-bawah.")
                            m2.metric("Corners (35%)", f"{grade_result['corners']['score']:.1f}",
                                    help="Kondisi sudut kartu.")
                            m3.metric("Edge Wear (25%)", f"{grade_result['edge_wear']['score']:.1f}",
                                    help="Kondisi tepi kartu.")

                            with st.expander("📐 Detail Centering Border", expanded=False):
                                c1, c2 = st.columns(2)
                                c1.metric("H-Symmetry", f"{grade_result['centering']['h_symmetry']:.1f}%")
                                c2.metric("V-Symmetry", f"{grade_result['centering']['v_symmetry']:.1f}%")
                                if centering_note != "ok":
                                    st.caption(f"ℹ️ {centering_note.replace('_',' ')} — skor centering menggunakan estimasi 75.0.")
                                else:
                                    st.caption(
                                        f"Border: Kiri {grade_result['centering']['left_border']}px | "
                                        f"Kanan {grade_result['centering']['right_border']}px | "
                                        f"Atas {grade_result['centering']['top_border']}px | "
                                        f"Bawah {grade_result['centering']['bottom_border']}px"
                                    )

                            with st.expander("🔲 Detail Kondisi Corners", expanded=False):
                                r1, r2 = st.columns(2)
                                r1.metric("↖ Top-Left",    f"{grade_result['corners']['top_left']:.1f}")
                                r2.metric("↗ Top-Right",   f"{grade_result['corners']['top_right']:.1f}")
                                r3, r4 = st.columns(2)
                                r3.metric("↙ Bottom-Left", f"{grade_result['corners']['bottom_left']:.1f}")
                                r4.metric("↘ Bottom-Right",f"{grade_result['corners']['bottom_right']:.1f}")

                            with st.expander("📏 Detail Edge Wear", expanded=False):
                                e1, e2 = st.columns(2)
                                e1.metric("⬆ Top",    f"{grade_result['edge_wear']['top']:.1f}")
                                e2.metric("⬇ Bottom", f"{grade_result['edge_wear']['bottom']:.1f}")
                                e3, e4 = st.columns(2)
                                e3.metric("⬅ Left",   f"{grade_result['edge_wear']['left']:.1f}")
                                e4.metric("➡ Right",  f"{grade_result['edge_wear']['right']:.1f}")

                    except Exception as e:
                        st.error(f"Error pada File #{idx+1}: {e}")
                        import traceback; st.code(traceback.format_exc())

            for t in tmp_paths:
                if os.path.exists(t): os.remove(t)
            st.toast("Seluruh gambar selesai dianalisis!", icon="🎉")

        else:
            st.markdown("""
                <div style="text-align:center;padding:50px 20px;border:2px dashed rgba(128,128,128,0.3);border-radius:12px;background:rgba(128,128,128,0.05);">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
                        width="110" style="opacity:0.35;margin-bottom:15px;"/>
                    <h5 style="font-weight:600;margin-bottom:5px;">Sistem PokéScan Siap</h5>
                    <p style="font-size:0.85rem;max-width:320px;margin:0 auto;opacity:0.7;">
                        Tekan tombol di atas untuk memulai analisis kartu.
                    </p>
                </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;padding:70px 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
                width="130" style="opacity:0.15;margin-bottom:20px;"/>
            <h4 style="font-weight:500;opacity:0.8;">Menunggu Input Gambar Kartu</h4>
            <p style="font-size:0.9rem;opacity:0.7;">
                Gunakan menu <b>Sidebar kiri</b> untuk memilih upload file atau kamera.
            </p>
        </div>
    """, unsafe_allow_html=True)