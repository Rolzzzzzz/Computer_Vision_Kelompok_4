import os
import sys
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
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

# PERBAIKAN: C dan gamma sebelumnya di-hardcode (C=10, gamma="scale") tanpa
# divalidasi. Pada feature space berdimensi tinggi, gamma="scale" sangat
# rentan membuat kernel RBF "vanish" untuk data baru yang sedikit berbeda
# dari training set -> decision_function selalu jatuh ke nilai bias/intercept
# (model jadi buta terhadap input, walau kelihatan "jalan normal").
# Sekarang nilai-nilai ini dicari otomatis lewat cross-validation supaya
# kombinasi yang benar-benar bisa generalisasi yang dipilih, bukan ditebak.
SVM_PARAM_GRID = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001, 0.0001]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
]

USE_PCA = True
PCA_VARIANCE = 0.95
# PERBAIKAN: sebelumnya PCA hanya dibatasi target variance (0.95) tanpa batas
# atas jumlah komponen. Pada dataset kecil, ini bisa menghasilkan PCA dengan
# komponen yang jumlahnya mendekati (atau lebih besar dari) jumlah sampel
# training itu sendiri -> SVM overfit parah & tidak generalisasi ke gambar
# baru. PCA_MAX_COMPONENTS membatasi jumlah komponen secara proporsional
# terhadap jumlah data training yang sebenarnya tersedia.
PCA_MAX_COMPONENTS_RATIO = 1 / 3   # maksimum ~1/3 dari jumlah sampel training
PCA_MAX_COMPONENTS_HARD_CAP = 120
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
        pca_probe = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
        pca_probe.fit(X_train_scaled)
        n_components = pca_probe.n_components_

        # PERBAIKAN: batasi jumlah komponen PCA relatif terhadap jumlah
        # sampel training, supaya dimensi fitur akhir tidak jauh lebih besar
        # daripada jumlah data yang ada (penyebab utama model "menghafal"
        # training set lalu buta terhadap gambar baru).
        n_components_cap = max(
            2,
            min(
                n_components,
                int(len(y_train) * PCA_MAX_COMPONENTS_RATIO),
                PCA_MAX_COMPONENTS_HARD_CAP,
            )
        )
        if n_components_cap < n_components:
            print(f"  [!] PCA awal butuh {n_components} komponen untuk variance={PCA_VARIANCE}, "
                  f"tapi itu terlalu besar dibanding {len(y_train)} data training.")
            print(f"      Membatasi jumlah komponen PCA menjadi {n_components_cap} untuk mengurangi risiko overfitting.")
        pca = PCA(n_components=n_components_cap, random_state=RANDOM_STATE)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"  Dimensi Fitur Menyusut menjadi: {X_train_scaled.shape[1]} "
              f"(variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

    print("\n[4/6] Melatih Model Klasifikasi Klasik (SVM) ...")
    t2 = time.time()
    cv_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_svm = SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base_svm, SVM_PARAM_GRID, scoring="f1", cv=cv_search, n_jobs=-1, refit=True
    )
    grid.fit(X_train_scaled, y_train)
    svm = grid.best_estimator_
    print(f"  Hyperparameter terbaik (CV): {grid.best_params_}  (CV F1={grid.best_score_*100:.2f}%)")
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