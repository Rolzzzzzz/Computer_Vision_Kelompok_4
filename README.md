# ⚡ PokéScan — Pokémon Card Fake/Real Detector & Grader

PokéScan adalah sistem berbasis **Computer Vision tradisional dan Machine Learning (SVM)** yang dirancang untuk menganalisis kartu Pokémon secara otomatis tanpa menggunakan *Deep Learning*. Aplikasi ini dapat membedakan antara kartu asli dan palsu (autentikasi) sekaligus mengevaluasi kondisi fisik kartu tersebut (*grading*).

Dibangun menggunakan **Streamlit**, aplikasi ini menyediakan antarmuka interaktif yang sangat responsif dan mendukung secara penuh mode gelap maupun terang (*Dark/Light Mode*) menyesuaikan dengan preferensi perangkat pengguna.

---

## ✨ Fitur Utama
* 🔍 **Autentikasi Kartu (Real vs Fake):** Ekstraksi fitur menggunakan HOG (*Histogram of Oriented Gradients*), LBP (*Local Binary Patterns*), Color Histograms, dan ORB yang diklasifikasikan dengan Support Vector Machine (SVM).
* 📏 **Sistem Grading Otomatis:** Mengevaluasi kondisi fisik kartu berdasarkan 3 komponen:
  * **Border Centering (40%):** Proyeksi horizontal/vertikal untuk mengukur skor simetri kartu.
  * **Corner Damage (35%):** Mengukur variansi Laplacian pada 4 sudut kartu.
  * **Edge Wear (25%):** Mengukur magnitudo gradien Sobel pada tepi (*strip*) kartu.
* 🚀 **Batch Processing:** Mendukung pengunggahan dan pemrosesan hingga **10 gambar sekaligus** dengan hasil yang ditampilkan rapi dalam bentuk *collapsible cards* lengkap dengan nama file aslinya.
* 📸 **Input Fleksibel:** Mendukung pengambilan gambar dari **Webcam** perangkat atau fitur **Upload File**.

---

## 📊 Hasil Evaluasi Model (Training)
Model SVM (*kernel=rbf*) dievaluasi pada data uji dan mendapatkan performa yang sangat solid:

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.90% |
| **Precision** | 98.39% |
| **Recall** | 100.00% |
| **F1-Score** | 99.19% |
| **CV F1 (5-fold)** | 98.12% ± 1.21% |

### Klasifikasi Grade Fisik
| Grade | Score | Keterangan |
|-------|-------|------------|
| **Mint** | 90–100 | Hampir sempurna, seperti baru, nyaris tanpa cacat. |
| **Near Mint** | 70–89 | Kondisi sangat baik, sedikit tanda pemakaian minor. |
| **Good** | 45–69 | Kondisi sedang, terlihat bekas dipakai secara jelas. |
| **Poor** | 0–44 | Rusak signifikan, banyak keausan. |

---

## 📂 Struktur Proyek
Struktur di bawah ini adalah file inti yang digunakan pada tahap produksi/deployment:

```text
.
├── models/                   # Folder berisi model yang sudah dilatih (pkl)
│   ├── pokemon_svm_model.pkl
│   └── pokemon_svm_scaler.pkl
├── app.py                    # Script antarmuka aplikasi utama (Streamlit UI)
├── feature_extraction.py     # Modul ekstraksi HOG, LBP, Color, ORB
├── grading.py                # Algoritma dan logika perhitungan nilai/kondisi
├── predict.py                # Script CLI murni (tanpa UI)
├── preprocessing.py          # Tahapan prapemrosesan (Crop, Blur, Grayscale, dll)
├── train.py                  # Pipeline pelatihan model machine learning
├── utils.py                  # Fungsi utilitas & visualisasi
├── requirements.txt          # Daftar spesifikasi library Python (menggunakan headless)
├── .gitignore                # Aturan pengecualian file untuk Git/GitHub
└── README.md                 # Dokumentasi proyek
