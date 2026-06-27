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

def analyze_border_centering(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h_proj = np.sum(thresh, axis=0).astype(float)
    v_proj = np.sum(thresh, axis=1).astype(float)

    h_proj_norm = h_proj / h_proj.max() if h_proj.max() > 0 else h_proj
    v_proj_norm = v_proj / v_proj.max() if v_proj.max() > 0 else v_proj

    threshold_val = 0.3

    h_active = np.where(h_proj_norm > threshold_val)[0]
    v_active = np.where(v_proj_norm > threshold_val)[0]

    if len(h_active) < 2 or len(v_active) < 2:
        return {
            "left": 0, "right": 0, "top": 0, "bottom": 0,
            "h_symmetry": 50, "v_symmetry": 50, "centering_score": 50
        }

    left = int(h_active[0])
    right = int(w - h_active[-1])
    top = int(v_active[0])
    bottom = int(h - v_active[-1])

    def symmetry_score(a: int, b: int) -> float:
        if a + b == 0:
            return 100.0
        diff_ratio = abs(a - b) / (a + b)
        return max(0.0, 100.0 * (1 - 2 * diff_ratio))

    h_sym = symmetry_score(left, right)
    v_sym = symmetry_score(top, bottom)
    centering = (h_sym + v_sym) / 2

    return {
        "left": left, "right": right, "top": top, "bottom": bottom,
        "h_symmetry": round(h_sym, 1),
        "v_symmetry": round(v_sym, 1),
        "centering_score": round(centering, 1),
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
    corners = analyze_corners(gray, corner_size=25)
    edge_wear = analyze_edge_wear(gray, edge_width=10)

    centering_score = centering["centering_score"]
    corner_score = corners["damage_score"]
    wear_score = edge_wear["wear_score"]

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
            "left_border": centering["left"],
            "right_border": centering["right"],
            "top_border": centering["top"],
            "bottom_border": centering["bottom"],
            "h_symmetry": centering["h_symmetry"],
            "v_symmetry": centering["v_symmetry"],
        },
        "corners": {
            "score": corner_score,
            "top_left": corners["top_left"],
            "top_right": corners["top_right"],
            "bottom_left": corners["bottom_left"],
            "bottom_right": corners["bottom_right"],
        },
        "edge_wear": {
            "score": wear_score,
            "top": edge_wear["top"],
            "bottom": edge_wear["bottom"],
            "left": edge_wear["left"],
            "right": edge_wear["right"],
        },
    }


def print_grade_report(result: dict, image_name: str = "Card") -> None:
    w = 50
    print("\n" + "═" * w)
    print(f"  POKÉMON CARD GRADING REPORT: {image_name}")
    print("═" * w)
    print(f"  Grade        : {result['grade']}")
    print(f"  Total Score  : {result['total_score']}/100")
    print("─" * w)
    print(f"  Centering    : {result['centering']['score']:.1f}/100 (bobot 40%)")
    print(f"    H-Symmetry : {result['centering']['h_symmetry']:.1f}%")
    print(f"    V-Symmetry : {result['centering']['v_symmetry']:.1f}%")
    print(f"  Corners      : {result['corners']['score']:.1f}/100 (bobot 35%)")
    print(f"  Edge Wear    : {result['edge_wear']['score']:.1f}/100 (bobot 25%)")
    print("═" * w)
