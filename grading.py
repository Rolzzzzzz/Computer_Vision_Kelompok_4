import cv2
import numpy as np

GRADES = {
    "Mint":      (90, 100),
    "Near Mint": (70, 89),
    "Good":      (45, 69),
    "Poor":      (0,  44),
}

def score_to_grade(score: float) -> str:
    for grade, (lo, hi) in GRADES.items():
        if lo <= score <= hi:
            return grade
    return "Poor"

def preprocess_for_grading(img: np.ndarray) -> tuple:
    img_resized = cv2.resize(img, (300, 420), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    return img_resized, gray, edges

def find_card_contour(edges: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = img_shape[:2]
    min_area = (h * w) * 0.3
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    x, y, w_r, h_r = cv2.boundingRect(contours[0])
    rect = np.array([
        [[x, y]], [[x + w_r, y]], [[x + w_r, y + h_r]], [[x, y + h_r]]
    ])
    return rect


def _find_card_bbox(img: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours_sorted[:5]:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.20:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.98 and bh > h * 0.98:
                continue
            return (x, y, x + bw, y + bh)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours2:
        largest = max(contours2, key=cv2.contourArea)
        if cv2.contourArea(largest) > h * w * 0.15:
            x, y, bw, bh = cv2.boundingRect(largest)
            if not (bw > w * 0.98 and bh > h * 0.98):
                return (x, y, x + bw, y + bh)

    sobelx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_thresh = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)

    col_sum = np.sum(grad_thresh, axis=0)
    row_sum = np.sum(grad_thresh, axis=1)

    col_threshold = col_sum.max() * 0.05 if col_sum.max() > 0 else 1
    row_threshold = row_sum.max() * 0.05 if row_sum.max() > 0 else 1

    col_active = np.where(col_sum > col_threshold)[0]
    row_active = np.where(row_sum > row_threshold)[0]

    if len(col_active) >= 2 and len(row_active) >= 2:
        x1, x2 = int(col_active[0]), int(col_active[-1])
        y1, y2 = int(row_active[0]), int(row_active[-1])
        bw, bh = x2 - x1, y2 - y1
        if bw > w * 0.05 and bh > h * 0.05:
            return (x1, y1, x2, y2)

    return None


def analyze_border_centering(img: np.ndarray) -> dict:
    h, w = img.shape[:2]

    bbox = _find_card_bbox(img)
    if bbox is None:
        return {
            "left": 0, "right": 0, "top": 0, "bottom": 0,
            "h_symmetry": 75.0, "v_symmetry": 75.0, "centering_score": 75.0,
            "note": "no_card_detected"
        }

    x1, y1, x2, y2 = bbox
    card_w = x2 - x1
    card_h = y2 - y1

    if card_w > w * 0.90 and card_h > h * 0.90:
        return {
            "left": 0, "right": 0, "top": 0, "bottom": 0,
            "h_symmetry": 75.0, "v_symmetry": 75.0, "centering_score": 75.0,
            "note": "card_fills_frame"
        }

    left   = max(0, x1)
    right  = max(0, w - x2)
    top    = max(0, y1)
    bottom = max(0, h - y2)

    def symmetry_score(a: int, b: int) -> float:
        if a + b == 0:
            return 75.0
        diff_ratio = abs(a - b) / (a + b)
        return max(0.0, 100.0 * (1.0 - 2.0 * diff_ratio))

    h_sym = symmetry_score(left, right)
    v_sym = symmetry_score(top, bottom)
    centering = (h_sym + v_sym) / 2.0

    return {
        "left": left, "right": right, "top": top, "bottom": bottom,
        "h_symmetry": round(h_sym, 1),
        "v_symmetry": round(v_sym, 1),
        "centering_score": round(centering, 1),
        "note": "ok"
    }


def analyze_corners(gray: np.ndarray, corner_size: int = 20) -> dict:
    h, w = gray.shape
    cs = min(corner_size, h // 6, w // 6)

    corners = {
        "top_left":     gray[0:cs, 0:cs],
        "top_right":    gray[0:cs, w-cs:w],
        "bottom_left":  gray[h-cs:h, 0:cs],
        "bottom_right": gray[h-cs:h, w-cs:w],
    }

    corner_scores = {}
    for name, patch in corners.items():
        if patch.size == 0:
            corner_scores[name] = 50.0
            continue
        lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F)
        edge_variance = lap.var()
        pixel_std = patch.std()
        damage_indicator = min(100.0, edge_variance / 2.0 + pixel_std)
        corner_score = max(0.0, 100.0 - damage_indicator)
        corner_scores[name] = round(corner_score, 1)

    avg_corner_score = np.mean(list(corner_scores.values()))
    consistency_penalty = np.std(list(corner_scores.values()))
    damage_score = max(0.0, avg_corner_score - consistency_penalty * 0.5)

    return {
        **corner_scores,
        "damage_score": round(damage_score, 1),
    }


def analyze_edge_wear(gray: np.ndarray, edge_width: int = 8) -> dict:
    h, w = gray.shape
    ew = min(edge_width, h // 10, w // 10)

    strips = {
        "top":    gray[0:ew, :],
        "bottom": gray[h-ew:h, :],
        "left":   gray[:, 0:ew],
        "right":  gray[:, w-ew:w],
    }

    wear_scores = {}
    for name, strip in strips.items():
        if strip.size == 0:
            wear_scores[name] = 70.0
            continue
        sobelx = cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        roughness = magnitude.mean()
        wear_score = max(0.0, 100.0 - roughness * 2.5)
        wear_scores[name] = round(wear_score, 1)

    avg_wear_score = np.mean(list(wear_scores.values()))

    return {
        **wear_scores,
        "wear_score": round(avg_wear_score, 1),
    }


def grade_card(img: np.ndarray) -> dict:
    img_resized, gray, edges = preprocess_for_grading(img)

    centering = analyze_border_centering(img_resized)
    corners   = analyze_corners(gray, corner_size=25)
    edge_wear = analyze_edge_wear(gray, edge_width=10)

    centering_score = centering["centering_score"]
    corner_score    = corners["damage_score"]
    wear_score      = edge_wear["wear_score"]

    total_score = (
        0.40 * centering_score +
        0.35 * corner_score +
        0.25 * wear_score
    )
    total_score = round(total_score, 1)
    grade = score_to_grade(total_score)

    return {
        "grade": grade,
        "total_score": total_score,
        "centering": {
            "score": centering_score,
            "left_border":  centering["left"],
            "right_border": centering["right"],
            "top_border":   centering["top"],
            "bottom_border": centering["bottom"],
            "h_symmetry":   centering["h_symmetry"],
            "v_symmetry":   centering["v_symmetry"],
            "note":         centering.get("note", "ok"),
        },
        "corners": {
            "score":        corner_score,
            "top_left":     corners["top_left"],
            "top_right":    corners["top_right"],
            "bottom_left":  corners["bottom_left"],
            "bottom_right": corners["bottom_right"],
        },
        "edge_wear": {
            "score":  wear_score,
            "top":    edge_wear["top"],
            "bottom": edge_wear["bottom"],
            "left":   edge_wear["left"],
            "right":  edge_wear["right"],
        },
    }


def print_grade_report(result: dict, image_name: str = "Card") -> None:
    w = 50
    note = result["centering"].get("note", "ok")
    centering_note = ""
    if note == "card_fills_frame":
        centering_note = " (kartu memenuhi frame, estimasi)"
    elif note == "no_card_detected":
        centering_note = " (kartu tidak terdeteksi, estimasi)"

    print("\n" + "═" * w)
    print(f"  POKÉMON CARD GRADING REPORT: {image_name}")
    print("═" * w)
    print(f"  Grade        : {result['grade']}")
    print(f"  Total Score  : {result['total_score']}/100")
    print("─" * w)
    print(f"  Centering    : {result['centering']['score']:.1f}/100 (bobot 40%){centering_note}")
    print(f"    H-Symmetry : {result['centering']['h_symmetry']:.1f}%")
    print(f"    V-Symmetry : {result['centering']['v_symmetry']:.1f}%")
    print(f"  Corners      : {result['corners']['score']:.1f}/100 (bobot 35%)")
    print(f"  Edge Wear    : {result['edge_wear']['score']:.1f}/100 (bobot 25%)")
    print("═" * w)
