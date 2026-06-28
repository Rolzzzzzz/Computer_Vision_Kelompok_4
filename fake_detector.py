"""
fake_detector.py — Visual-based fake card detector.

Dipakai sebagai sistem utama ketika model SVM tidak bisa membedakan
REAL vs FAKE (misalnya karena model broken/undertrained).

Kartu Pokémon asli punya ciri khas:
- Print quality tinggi: detail tajam, tidak blur
- Warna saturasi kuat dan konsisten
- Pola dot-print (halftone) reguler di bawah magnifikasi
- Teks cetak tajam (edge pada teks sangat sharp)
- Hologram di kartu rarity tinggi
- Tekstur kertas khusus (tidak bisa dideteksi via foto, skip)
- Warna border/background konsisten dengan set aslinya

Kartu FAKE biasanya punya:
- Print blur / washed out
- Warna terlalu jenuh (oversaturated) atau terlalu pucat
- Teks blur di tepi
- Pola halftone tidak reguler atau terlalu kasar
- Noise/artifact kompresi berlebihan
- Border tidak presisi
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ─────────────────────────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────────

def _sharpness_score(gray: np.ndarray) -> float:
    """
    Laplacian variance sebagai ukuran ketajaman gambar.
    Kartu asli → sharp → variance tinggi.
    Kartu fake sering blur → variance rendah.
    """
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _print_quality_score(gray: np.ndarray) -> float:
    """
    Analisis kualitas cetak berdasarkan local contrast.
    Kartu asli punya high-frequency detail yang kaya.
    """
    # Sobel magnitude
    sx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return float(mag.mean())


def _color_consistency_score(img_bgr: np.ndarray) -> dict:
    """
    Analisis distribusi warna.
    Kartu fake sering punya distribusi warna yang ekstrem (terlalu saturasi
    atau terlalu pucat) dan kurang konsisten antar region.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    return {
        "sat_mean":   float(s.mean()),
        "sat_std":    float(s.std()),
        "val_mean":   float(v.mean()),
        "val_std":    float(v.std()),
        "hue_std":    float(h.std()),
    }


def _text_region_sharpness(gray: np.ndarray) -> float:
    """
    Fokus pada area teks (region dengan kontras tinggi tapi area kecil).
    Teks pada kartu asli sangat tajam.
    """
    # Ambil bagian bawah kartu (area nama, HP, teks deskripsi)
    h, w = gray.shape
    text_region = gray[int(h*0.6):, :]

    lap = cv2.Laplacian(text_region.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _noise_level(gray: np.ndarray) -> float:
    """
    Estimasi noise level. Kartu fake dari foto/scan kartu palsu sering
    punya kompresi artifact → noise tertentu.
    """
    # Blur lalu ambil residual sebagai noise estimate
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise = gray.astype(np.float32) - blurred
    return float(noise.std())


def _lbp_uniformity(gray: np.ndarray) -> float:
    """
    LBP texture uniformity. Kartu asli punya pola tekstur yang khas
    (kertas khusus + cetak offset). Kartu fake (inkjet/laser print) 
    punya distribusi LBP berbeda.
    Kita ukur seberapa 'uniform' distribusinya — kartu asli lebih terstruktur.
    """
    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")
    n_bins = 18  # P+2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # Entropy dari distribusi LBP
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return float(entropy)


def _halftone_regularity(gray: np.ndarray) -> float:
    """
    Deteksi pola halftone (dot pattern) menggunakan FFT.
    Kartu asli menggunakan cetak offset yang menghasilkan pola
    halftone reguler pada frekuensi tertentu.
    Foto kartu fake dari printer rumahan tidak punya pola ini.
    
    Returns: peak energy di mid-frequency band (normalized)
    """
    # Crop bagian tengah (menghindari border dan efek tepi)
    h, w = gray.shape
    crop = gray[h//4:3*h//4, w//4:3*w//4]

    # FFT
    f = np.fft.fft2(crop.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    ch, cw = magnitude.shape
    cy, cx = ch // 2, cw // 2

    # Low frequency (0-10% radius) — dominated by DC component, skip
    # Mid frequency (10-40% radius) — halftone pattern lives here
    # High frequency (40%+) — noise
    r_max = min(cy, cx)
    Y, X = np.ogrid[:ch, :cw]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    mid_mask = (dist >= r_max * 0.10) & (dist <= r_max * 0.40)
    total_energy = magnitude.sum() + 1e-10
    mid_energy = magnitude[mid_mask].sum() / total_energy

    return float(mid_energy)


def _border_regularity(img_bgr: np.ndarray) -> float:
    """
    Cek ketepatan border kartu. Kartu asli punya border yang sangat
    presisi dan warna solid. Kartu fake sering punya border yang sedikit
    off atau warnanya tidak konsisten.
    """
    h, w = img_bgr.shape[:2]
    ew = max(5, min(15, h // 30))

    # Ambil strip border
    borders = [
        img_bgr[0:ew, :],          # top
        img_bgr[h-ew:h, :],        # bottom
        img_bgr[:, 0:ew],          # left
        img_bgr[:, w-ew:w],        # right
    ]

    # Konsistensi warna dalam border (std rendah = warna solid = bagus)
    std_scores = []
    for b in borders:
        if b.size == 0:
            continue
        std_scores.append(b.std())

    if not std_scores:
        return 50.0

    # Semakin rendah std, semakin konsisten border → lebih mungkin asli
    # Normalize: std=0 → score=100, std=60+ → score=0
    avg_std = np.mean(std_scores)
    score = max(0.0, 100.0 - avg_std * 1.5)
    return float(score)


# ─────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────

def analyze_authenticity(img_bgr: np.ndarray) -> dict:
    """
    Analisis keaslian kartu berdasarkan fitur visual.
    
    Returns dict dengan:
    - prediction: 0 = REAL, 1 = FAKE
    - confidence: float 0–1
    - score_real: float 0–100
    - details: dict berisi skor tiap komponen
    """
    target_size = (300, 420)
    img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ekstrak semua fitur
    sharpness      = _sharpness_score(gray)
    print_quality  = _print_quality_score(gray)
    color_info     = _color_consistency_score(img)
    text_sharp     = _text_region_sharpness(gray)
    noise          = _noise_level(gray)
    lbp_entropy    = _lbp_uniformity(gray)
    halftone       = _halftone_regularity(gray)
    border_reg     = _border_regularity(img)

    # ── Scoring per komponen (0–100, lebih tinggi = lebih likely REAL) ──

    # 1. Sharpness: kartu asli biasanya lap_var > 300
    #    Batas bawah ~50 (blur), batas atas ~2000+ (sangat tajam)
    sharp_score = min(100.0, max(0.0, (sharpness - 50) / (800 - 50) * 100))

    # 2. Print quality (edge mean): asli ~10–30, fake ~5–10 atau >35 (artifact)
    if print_quality < 5:
        pq_score = 20.0
    elif print_quality > 40:
        pq_score = 60.0  # Terlalu tinggi juga bisa artifact
    else:
        pq_score = min(100.0, (print_quality - 5) / (30 - 5) * 100)

    # 3. Text sharpness: asli biasanya lap_var > 200 di region teks
    text_score = min(100.0, max(0.0, (text_sharp - 30) / (600 - 30) * 100))

    # 4. Color saturation: kartu asli saturasi 80–160, fake sering < 60 atau > 200
    sat = color_info["sat_mean"]
    if sat < 40:
        color_score = 20.0   # terlalu pucat → fake
    elif sat > 220:
        color_score = 30.0   # terlalu saturasi → fake
    elif 70 <= sat <= 170:
        color_score = 85.0   # range normal asli
    else:
        color_score = 55.0

    # 5. Noise: noise terlalu rendah (foto scan kartu sempurna) atau terlalu tinggi (gambar jelek)
    #    Range wajar untuk foto kartu asli: 3–12
    if noise < 2:
        noise_score = 40.0   # suspiciously clean → mungkin digital fake
    elif noise > 20:
        noise_score = 30.0   # terlalu noisy → kualitas buruk
    elif 3 <= noise <= 12:
        noise_score = 85.0
    else:
        noise_score = 60.0

    # 6. LBP entropy: kartu asli punya entropy ~3.5–4.2 (texture kaya tapi terstruktur)
    #    Kartu fake (gambar digital murni atau laser print) biasanya entropy < 3.0 atau > 4.5
    if lbp_entropy < 2.8:
        lbp_score = 25.0
    elif lbp_entropy > 4.5:
        lbp_score = 40.0
    elif 3.2 <= lbp_entropy <= 4.2:
        lbp_score = 85.0
    else:
        lbp_score = 60.0

    # 7. Halftone regularity: kartu asli punya mid-freq energy yang lebih tinggi
    #    (pola cetak offset). Range khas: 0.25–0.45 untuk asli.
    if halftone < 0.20:
        halftone_score = 30.0  # tidak ada pola halftone → digital/fake
    elif halftone > 0.50:
        halftone_score = 50.0  # terlalu banyak mid-freq → artifact
    elif 0.25 <= halftone <= 0.45:
        halftone_score = 80.0
    else:
        halftone_score = 60.0

    # 8. Border regularity: asli punya border solid dan konsisten
    border_score = border_reg

    # ── Weighted average ──
    weights = {
        "sharpness":   0.20,
        "print":       0.15,
        "text":        0.15,
        "color":       0.15,
        "noise":       0.10,
        "lbp":         0.10,
        "halftone":    0.10,
        "border":      0.05,
    }
    scores = {
        "sharpness":   sharp_score,
        "print":       pq_score,
        "text":        text_score,
        "color":       color_score,
        "noise":       noise_score,
        "lbp":         lbp_score,
        "halftone":    halftone_score,
        "border":      border_score,
    }
    total = sum(weights[k] * scores[k] for k in weights)
    score_real = round(total, 1)

    # Threshold: >= 55 → REAL, < 55 → FAKE
    THRESHOLD = 55.0
    prediction = 0 if score_real >= THRESHOLD else 1
    confidence = score_real / 100.0 if prediction == 0 else (100.0 - score_real) / 100.0
    confidence = round(min(0.99, max(0.50, confidence)), 3)

    return {
        "prediction":  prediction,
        "confidence":  confidence,
        "score_real":  score_real,
        "threshold":   THRESHOLD,
        "details": {
            "sharpness_raw":    round(sharpness, 2),
            "print_quality_raw": round(print_quality, 2),
            "text_sharp_raw":   round(text_sharp, 2),
            "sat_mean":         round(color_info["sat_mean"], 2),
            "sat_std":          round(color_info["sat_std"], 2),
            "noise_std":        round(noise, 2),
            "lbp_entropy":      round(lbp_entropy, 3),
            "halftone_energy":  round(halftone, 4),
            **scores,
        }
    }
