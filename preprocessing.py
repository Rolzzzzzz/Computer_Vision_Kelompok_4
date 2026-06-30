import os
import csv
import cv2
import numpy as np

TARGET_SIZE = (128, 128)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Gagal load: {path}")
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

    img_color = resize_image(img, TARGET_SIZE)
    gray = to_grayscale(img_color)
    blurred = apply_gaussian_blur(gray, kernel_size=3)
    equalized = equalize_histogram(blurred)

    return img_color, equalized


def load_dataset_from_csv(img_dir: str, csv_path: str) -> tuple[list, list, list, list]:
    paths = []
    color_images = []
    gray_images = []
    labels = []

    if not os.path.exists(csv_path):
        print(f"ERROR! File CSV tidak ditemukan: {csv_path}")
        return paths, color_images, gray_images, labels

    csv_name = os.path.basename(csv_path)
    print(f"  Memproses gambar dari folder '{os.path.basename(img_dir)}' berdasarkan file '{csv_name}'...")
    
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            img_id = row['id']
            label = int(row['label'])
            
            filename = f"{int(img_id)}.JPG" 
            fpath = os.path.join(img_dir, filename)
            
            if not os.path.exists(fpath):
                fpath = os.path.join(img_dir, f"{int(img_id)}.jpg")

            if not os.path.exists(fpath):
                print(f"WARNING! Gambar tidak ditemukan di disk: {fpath}")
                continue

            img_color, img_gray = preprocess_image(fpath)
            if img_color is None:
                continue

            paths.append(fpath)
            color_images.append(img_color)
            gray_images.append(img_gray)
            labels.append(label)

    print(f"  Berhasil memuat {len(labels)} gambar dari {csv_name}.")
    return paths, color_images, gray_images, labels