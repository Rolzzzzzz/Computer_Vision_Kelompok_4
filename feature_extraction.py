import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_hog(gray: np.ndarray) -> np.ndarray:
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features.astype(np.float32)

def extract_lbp(gray: np.ndarray, n_points: int = 24, radius: int = 3) -> np.ndarray:
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def extract_color_histogram(img_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    ranges = [(0, 180), (0, 256), (0, 256)]

    for ch_idx, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([hsv], [ch_idx], None, [bins], [lo, hi])
        hist = hist.flatten()
        total = hist.sum()
        if total > 0:
            hist = hist / total
        features.append(hist)

    return np.concatenate(features).astype(np.float32)

def extract_orb_stats(gray: np.ndarray, n_keypoints: int = 200) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=n_keypoints)
    keypoints, _ = orb.detectAndCompute(gray, None)

    if len(keypoints) == 0:
        return np.zeros(9, dtype=np.float32)

    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)

    features = np.array([
        len(keypoints),
        responses.mean(),
        responses.std(),
        responses.max(),
        sizes.mean(),
        sizes.std(),
        angles.mean(),
        angles.std(), 
        np.percentile(responses, 75) - np.percentile(responses, 25),
    ], dtype=np.float32)

    return features

def extract_all_features(img_color: np.ndarray, img_gray: np.ndarray) -> np.ndarray:
    hog_feat = extract_hog(img_gray)
    lbp_feat = extract_lbp(img_gray)
    color_feat = extract_color_histogram(img_color)
    orb_feat = extract_orb_stats(img_gray)

    feature_vector = np.concatenate([hog_feat, lbp_feat, color_feat, orb_feat])
    return feature_vector


def extract_features_batch(color_images: list, gray_images: list,
                        verbose: bool = True) -> np.ndarray:
    all_features = []
    n = len(color_images)

    for i, (img_c, img_g) in enumerate(zip(color_images, gray_images)):
        feat = extract_all_features(img_c, img_g)
        all_features.append(feat)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] feature extraction selesai ...")

    X = np.array(all_features, dtype=np.float32)
    if verbose:
        print(f"Feature matrix shape: {X.shape}")
    return X
