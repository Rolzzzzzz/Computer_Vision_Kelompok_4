import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (save to file)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import joblib

def save_model(model, pipeline: dict, model_dir: str, prefix: str = "pokemon_svm") -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{prefix}_model.pkl")
    scaler_path = os.path.join(model_dir, f"{prefix}_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(pipeline, scaler_path)
    print(f"[✓] Model disimpan: {model_path}")
    print(f"[✓] Pipeline (scaler+pca) disimpan: {scaler_path}")


def load_model(model_dir: str, prefix: str = "pokemon_svm"):
    model_path = os.path.join(model_dir, f"{prefix}_model.pkl")
    scaler_path = os.path.join(model_dir, f"{prefix}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Pipeline tidak ditemukan: {scaler_path}")

    model = joblib.load(model_path)
    pipeline = joblib.load(scaler_path)

    # backward compat: kalau pkl lama isinya cuma scaler (bukan dict)
    if not isinstance(pipeline, dict):
        pipeline = {"scaler": pipeline, "pca": None}

    print(f"[✓] Model dimuat dari: {model_path}")
    return model, pipeline

def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                        save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm.sum() * 100
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Confusion matrix disimpan: {save_path}")
    plt.close()

def plot_feature_groups(feature_vector: np.ndarray,
                        save_path: str | None = None) -> None:
    n = len(feature_vector)
    n_orb = 9
    n_color = 96
    n_lbp = 26
    n_hog = n - n_orb - n_color - n_lbp

    groups = {
        f"HOG ({n_hog}D)": feature_vector[:n_hog],
        f"LBP ({n_lbp}D)": feature_vector[n_hog:n_hog+n_lbp],
        f"Color ({n_color}D)": feature_vector[n_hog+n_lbp:n_hog+n_lbp+n_color],
        f"ORB ({n_orb}D)": feature_vector[-n_orb:],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig.suptitle("Feature Groups Distribution", fontsize=13, fontweight="bold")

    for ax, (name, feat) in zip(axes, groups.items()):
        ax.hist(feat, bins=40, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Feature distribution disimpan: {save_path}")
    plt.close()

def visualize_prediction(img_bgr: np.ndarray, prediction: int,
                        confidence: float, grade_result: dict,
                        save_path: str | None = None) -> None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")

    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    label_text = "✓ REAL" if prediction == 0 else "✗ FAKE"
    label_color = "#00e676" if prediction == 0 else "#ff5252"
    axes[0].set_title(label_text, fontsize=18, fontweight="bold",
                    color=label_color, pad=10)

    ax = axes[1]
    ax.set_facecolor("#16213e")
    ax.axis("off")

    grade = grade_result["grade"]
    total = grade_result["total_score"]
    grade_colors = {
        "Mint": "#ffd700",
        "Near Mint": "#00e676",
        "Good": "#ff9800",
        "Poor": "#ff5252",
    }
    gc = grade_colors.get(grade, "white")

    lines = [
        ("AUTHENTICATION", "#ffffff", 14),
        (f"{label_text}  ({confidence*100:.1f}%)", label_color, 20),
        ("", "#ffffff", 10),
        ("CARD GRADING", "#ffffff", 14),
        (f"Grade: {grade}", gc, 20),
        (f"Score: {total}/100", gc, 16),
        ("", "#ffffff", 10),
        (f"Centering : {grade_result['centering']['score']:.1f}", "#90caf9", 12),
        (f"Corners   : {grade_result['corners']['score']:.1f}", "#90caf9", 12),
        (f"Edge Wear : {grade_result['edge_wear']['score']:.1f}", "#90caf9", 12),
    ]

    y = 0.92
    for text, color, size in lines:
        ax.text(0.5, y, text, transform=ax.transAxes,
                ha="center", va="top", color=color, fontsize=size,
                fontweight="bold" if size >= 14 else "normal")
        y -= 0.09 + (size - 10) * 0.003

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Visualisasi disimpan: {save_path}")
    plt.close()

def plot_metrics_bar(metrics: dict, save_path: str | None = None) -> None:
    names = list(metrics.keys())
    values = [metrics[k] * 100 for k in names]
    colors = ["#2196f3", "#4caf50", "#ff9800", "#e91e63"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors[:len(names)], width=0.5, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Classification Metrics", fontsize=14, fontweight="bold")
    ax.axhline(100, linestyle="--", color="gray", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Metrics chart disimpan: {save_path}")
    plt.close()
