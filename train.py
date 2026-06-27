import os
import sys
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

from preprocessing import load_dataset_from_csv
from feature_extraction import extract_features_batch
from utils import save_model, plot_confusion_matrix, plot_metrics_bar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CLASS_NAMES = ["Real", "Fake"]

SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

USE_PCA = True
PCA_VARIANCE = 0.95
RANDOM_STATE = 42

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  POKÉMON CARD DETECTOR — TRAINING PIPELINE (MACHINE LEARNING)")
    print("=" * 60)

    train_dir = os.path.join(DATASET_DIR, "train")
    train_csv = os.path.join(DATASET_DIR, "train_labels.csv")
    test_dir = os.path.join(DATASET_DIR, "test")
    test_csv = os.path.join(DATASET_DIR, "test_labels.csv")

    print("\n[1/6] Loading Train & Test dataset ...")
    t0 = time.time()
    
    train_paths, train_color, train_gray, y_train = load_dataset_from_csv(train_dir, train_csv)
    test_paths, test_color, test_gray, y_test = load_dataset_from_csv(test_dir, test_csv)

    if len(y_train) == 0 or len(y_test) == 0:
        print("\n[ERROR] Dataset kosong! Pastikan file gambar ada di dalam folder train/test dan CSV-nya benar.")
        sys.exit(1)

    print(f"  [Selesai memuat dataset dalam {time.time() - t0:.2f} detik]")
    print(f"  Total Data: Train = {len(y_train)} gambar, Test = {len(y_test)} gambar")

    print("\n[2/6] Mengekstraksi Fitur (HOG, LBP, Color Hist, ORB) ...")
    t1 = time.time()
    print("  Ekstraksi fitur Data Train...")
    X_train = extract_features_batch(train_color, train_gray, verbose=False)
    print("  Ekstraksi fitur Data Test...")
    X_test = extract_features_batch(test_color, test_gray, verbose=False)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"  [Selesai ekstraksi fitur dalam {time.time() - t1:.2f} detik]")

    print("\n[3/6] Preprocessing (Scaling & PCA) ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if USE_PCA:
        print(f"  Mengaplikasikan PCA (Target Variance = {PCA_VARIANCE}) ...")
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"  Dimensi Fitur Menyusut menjadi: {X_train_scaled.shape[1]}")

    print("\n[4/6] Melatih Model Klasifikasi Klasik (SVM) ...")
    t2 = time.time()
    svm = SVC(
        C=SVM_C,
        kernel=SVM_KERNEL,
        gamma=SVM_GAMMA,
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE
    )
    svm.fit(X_train_scaled, y_train)
    print(f"  [Selesai training model dalam {time.time() - t2:.2f} detik]")

    print("\n[5/6] Evaluasi Model pada Data Test...")
    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "─" * 50)
    print("  HASIL EVALUASI (Test Set)")
    print("─" * 50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("─" * 50)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("  5-Fold Cross-Validation pada Train Data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"  CV F1: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    print("\n[6/6] Menyimpan Model & Visualisasi Hasil ...")
    pipeline_dict = {"scaler": scaler, "pca": pca}
    save_model(svm, pipeline_dict, MODEL_DIR, prefix="pokemon_svm")

    plot_confusion_matrix(cm, CLASS_NAMES, save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    plot_metrics_bar(metrics, save_path=os.path.join(RESULTS_DIR, "metrics.png"))
    
    print(f"\n[✓] Seluruh proses selesai! Model siap digunakan oleh predict.py")

if __name__ == "__main__":
    main()