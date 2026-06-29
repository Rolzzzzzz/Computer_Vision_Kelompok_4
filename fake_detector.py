import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def _sharpness_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _print_quality_score(gray: np.ndarray) -> float:
    sx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return float(mag.mean())


def _color_consistency_score(img_bgr: np.ndarray) -> dict:
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
    h, w = gray.shape
    text_region = gray[int(h*0.6):, :]

    lap = cv2.Laplacian(text_region.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _noise_level(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise = gray.astype(np.float32) - blurred
    return float(noise.std())


def _lbp_uniformity(gray: np.ndarray) -> float:
    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")
    n_bins = 18
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return float(entropy)


def _halftone_regularity(gray: np.ndarray) -> float:
    h, w = gray.shape
    crop = gray[h//4:3*h//4, w//4:3*w//4]

    f = np.fft.fft2(crop.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    ch, cw = magnitude.shape
    cy, cx = ch // 2, cw // 2

    r_max = min(cy, cx)
    Y, X = np.ogrid[:ch, :cw]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    mid_mask = (dist >= r_max * 0.10) & (dist <= r_max * 0.40)
    total_energy = magnitude.sum() + 1e-10
    mid_energy = magnitude[mid_mask].sum() / total_energy

    return float(mid_energy)


def _border_regularity(img_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    ew = max(5, min(15, h // 30))

    borders = [
        img_bgr[0:ew, :],
        img_bgr[h-ew:h, :],
        img_bgr[:, 0:ew],
        img_bgr[:, w-ew:w],
    ]

    std_scores = []
    for b in borders:
        if b.size == 0:
            continue
        std_scores.append(b.std())

    if not std_scores:
        return 50.0

    avg_std = np.mean(std_scores)
    score = max(0.0, 100.0 - avg_std * 1.5)
    return float(score)

def _detect_mirrored_text(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = img_bgr.shape[0]

    top_region = hsv[:h // 4, :]
    bot_region = hsv[3 * h // 4:, :]

    def _yellow_text_pct(region: np.ndarray) -> float:
        mask = (
            (region[:, :, 0] >= 18) & (region[:, :, 0] <= 32) &
            (region[:, :, 1] > 120) & (region[:, :, 2] > 120)
        )
        return float(mask.sum()) / (region.shape[0] * region.shape[1] + 1e-5)

    yp_top = _yellow_text_pct(top_region)
    yp_bot = _yellow_text_pct(bot_region)

    if yp_top > 0.01 and yp_bot > 0.01:
        balance = min(yp_top, yp_bot) / (max(yp_top, yp_bot) + 1e-5)
        suspicion = min(100.0, balance * 100.0 + 40.0)
    else:
        suspicion = 0.0

    return float(suspicion)


def _detect_solid_or_uniform_color(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_std = hsv[:, :, 1].std()
    val_std = hsv[:, :, 2].std()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    suspicion = 0.0
    if sat_std < 20 and lap_var < 100:
        suspicion = 90.0
    elif sat_std < 35 and lap_var < 200:
        suspicion = 60.0
    elif sat_std < 50 and lap_var < 400:
        suspicion = 30.0

    return float(suspicion)


def analyze_authenticity(img_bgr: np.ndarray) -> dict:
    target_size = (300, 420)
    img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness      = _sharpness_score(gray)
    print_quality  = _print_quality_score(gray)
    color_info     = _color_consistency_score(img)
    text_sharp     = _text_region_sharpness(gray)
    noise          = _noise_level(gray)
    lbp_entropy    = _lbp_uniformity(gray)
    halftone       = _halftone_regularity(gray)
    border_reg     = _border_regularity(img)
    mirror_suspicion = _detect_mirrored_text(img)
    solid_suspicion  = _detect_solid_or_uniform_color(img)

    sharp_score = min(100.0, max(0.0, (sharpness - 50) / (800 - 50) * 100))

    if print_quality < 5:
        pq_score = 20.0
    elif print_quality > 40:
        pq_score = 60.0
    else:
        pq_score = min(100.0, (print_quality - 5) / (30 - 5) * 100)

    text_score = min(100.0, max(0.0, (text_sharp - 30) / (600 - 30) * 100))

    sat = color_info["sat_mean"]
    if sat < 40:
        color_score = 20.0
    elif sat > 220:
        color_score = 30.0
    elif 70 <= sat <= 170:
        color_score = 85.0
    else:
        color_score = 55.0

    if noise < 2:
        noise_score = 40.0
    elif noise > 20:
        noise_score = 30.0
    elif 3 <= noise <= 12:
        noise_score = 85.0
    else:
        noise_score = 60.0

    if lbp_entropy < 2.8:
        lbp_score = 25.0
    elif lbp_entropy > 4.5:
        lbp_score = 40.0
    elif 3.2 <= lbp_entropy <= 4.2:
        lbp_score = 85.0
    else:
        lbp_score = 60.0

    if halftone < 0.20:
        halftone_score = 30.0
    elif halftone > 0.50:
        halftone_score = 50.0
    elif 0.25 <= halftone <= 0.45:
        halftone_score = 80.0
    else:
        halftone_score = 60.0

    border_score = border_reg

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
    if solid_suspicion >= 60.0:
        score_real = min(score_real, 30.0)
    if mirror_suspicion >= 50.0:
        score_real = min(score_real, 35.0)
    elif mirror_suspicion >= 25.0:
        penalty = mirror_suspicion * 0.3
        score_real = max(0.0, score_real - penalty)

    score_real = round(score_real, 1)

    THRESHOLD = 55.0
    prediction = 0 if score_real >= THRESHOLD else 1
    confidence = score_real / 100.0 if prediction == 0 else (100.0 - score_real) / 100.0
    confidence = round(min(0.99, max(0.50, confidence)), 3)

    return {
        "prediction":  prediction,
        "confidence":  confidence,
        "score_real":  score_real,
        "threshold":   THRESHOLD,
        "mirror_suspicion": round(mirror_suspicion, 1),
        "solid_suspicion":  round(solid_suspicion, 1),
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
