import os
import cv2
import numpy as np
import csv

TARGET_SIZE = (128, 176) 
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Gagal load: {path}")
    return img

def crop_card_safe(img: np.ndarray) -> np.ndarray:
    try:
        h, w = img.shape[:2]
        if max(h, w) < 200:
            return img
            
        scale = 400.0 / max(h, w)
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        card_contour = None
        
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area > (img_small.shape[0] * img_small.shape[1]) * 0.15:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    card_contour = approx
                    break
                    
        if card_contour is None:
            x, y, w_b, h_b = cv2.boundingRect(contours[0])
            if w_b * h_b > (img_small.shape[0] * img_small.shape[1]) * 0.25:
                card_contour = np.array([[[x, y]], [[x+w_b, y]], [[x+w_b, y+h_b]], [[x, y+h_b]]])
            else:
                return img
                
        card_contour = (card_contour / scale).astype(np.float32)
        pts = card_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   
        rect[2] = pts[np.argmax(s)]   
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth > maxHeight:
            maxWidth, maxHeight = maxHeight, maxWidth
            rect = np.array([tr, br, bl, tl], dtype="float32")
            
        dst = np.array([
            [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    except Exception:
        return img

def resize_image(img: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=0)

def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_image(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    img = load_image(path)
    if img is None:
        return None, None

    img_cropped = crop_card_safe(img)
    img_color = resize_image(img_cropped, TARGET_SIZE)
    
    gray = to_grayscale(img_color)
    blurred = apply_gaussian_blur(gray, kernel_size=3)
    equalized = equalize_histogram(blurred)

    return img_color, equalized

def load_dataset_from_csv(img_dir: str, csv_path: str) -> tuple[list, list, list, list]:
    paths, color_images, gray_images, labels = [], [], [], []

    if not os.path.exists(csv_path):
        print(f"[ERROR] File CSV tidak ditemukan: {csv_path}")
        return paths, color_images, gray_images, labels

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['id']
            label = int(row['label'])
            
            filename = f"{int(img_id)}.JPG" 
            fpath = os.path.join(img_dir, filename)
            if not os.path.exists(fpath):
                fpath = os.path.join(img_dir, f"{int(img_id)}.jpg")
            if not os.path.exists(fpath): continue

            img_color, img_gray = preprocess_image(fpath)
            if img_color is None: continue

            paths.append(fpath)
            color_images.append(img_color)
            gray_images.append(img_gray)
            labels.append(label)

    return paths, color_images, gray_images, labels