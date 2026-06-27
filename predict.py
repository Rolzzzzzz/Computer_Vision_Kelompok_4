import os
import sys
import argparse
import cv2
import numpy as np

from preprocessing import preprocess_image, SUPPORTED_EXT
from feature_extraction import extract_all_features
from grading import grade_card, print_grade_report
from utils import load_model, visualize_prediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CLASS_NAMES = {0: "REAL", 1: "FAKE"}
CLASS_EMOJI = {0: "✓", 1: "✗"}

def predict_card(image_path: str, model, pipeline: dict,
                do_grading: bool = True) -> dict:
    img_color, img_gray = preprocess_image(image_path)
    if img_color is None:
        return {"error": f"Gagal membaca gambar: {image_path}"}

    features = extract_all_features(img_color, img_gray)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    X = features.reshape(1, -1)
    X_scaled = scaler.transform(X)

    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    prediction = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(proba[prediction])

    grade_result = None
    if do_grading:
        img_orig = cv2.imread(image_path)
        if img_orig is not None:
            grade_result = grade_card(img_orig)

    return {
        "path": image_path,
        "prediction": prediction,
        "label": CLASS_NAMES[prediction],
        "confidence": confidence,
        "proba_real": float(proba[0]),
        "proba_fake": float(proba[1]),
        "grade_result": grade_result,
    }


def print_prediction(result: dict) -> None:
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    pred = result["prediction"]
    label = result["label"]
    conf = result["confidence"]
    emoji = CLASS_EMOJI[pred]
    color_tag = "\033[92m" if pred == 0 else "\033[91m"
    reset = "\033[0m"

    fname = os.path.basename(result["path"])
    print(f"\n{'─'*50}")
    print(f"  File       : {fname}")
    print(f"  Prediction : {color_tag}{emoji} {label}{reset}")
    print(f"  Confidence : {conf*100:.1f}%")
    print(f"  P(Real)    : {result['proba_real']*100:.1f}%")
    print(f"  P(Fake)    : {result['proba_fake']*100:.1f}%")

    if result.get("grade_result"):
        gr = result["grade_result"]
        print(f"  Grade      : {gr['grade']} ({gr['total_score']}/100)")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prediksi Real/Fake kartu Pokémon"
    )
    parser.add_argument("input", help="Path gambar atau folder")
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Simpan visualisasi hasil prediksi (PNG)"
    )
    parser.add_argument(
        "--no-grade", action="store_true",
        help="Skip card grading (lebih cepat)"
    )
    parser.add_argument(
        "--model-dir", default=MODEL_DIR,
        help=f"Folder model (default: {MODEL_DIR})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 50)
    print("  POKÉMON CARD DETECTOR — PREDICTION")
    print("=" * 50)

    print("\n[*] Loading model ...")
    try:
        model, pipeline = load_model(args.model_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("       Jalankan train.py terlebih dahulu!")
        sys.exit(1)

    do_grading = not args.no_grade

    input_path = args.input
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]
        if not files:
            print(f"[ERROR] Tidak ada gambar ditemukan di: {input_path}")
            sys.exit(1)
        print(f"[*] Memproses {len(files)} gambar dari folder ...")
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"[ERROR] Path tidak ditemukan: {input_path}")
        sys.exit(1)

    results = []
    for fpath in files:
        result = predict_card(fpath, model, pipeline, do_grading=do_grading)
        results.append(result)
        print_prediction(result)

        if args.visualize and "error" not in result and result.get("grade_result"):
            img = cv2.imread(fpath)
            if img is not None:
                save_name = os.path.splitext(os.path.basename(fpath))[0]
                save_path = os.path.join(RESULTS_DIR, f"result_{save_name}.png")
                visualize_prediction(
                    img, result["prediction"], result["confidence"],
                    result["grade_result"], save_path=save_path
                )

    if len(results) > 1:
        valid = [r for r in results if "error" not in r]
        n_real = sum(1 for r in valid if r["prediction"] == 0)
        n_fake = sum(1 for r in valid if r["prediction"] == 1)
        avg_conf = np.mean([r["confidence"] for r in valid]) if valid else 0
        print(f"\n{'═'*50}")
        print(f"  RINGKASAN: {len(valid)} gambar diproses")
        print(f"  Real: {n_real}  |  Fake: {n_fake}")
        print(f"  Rata-rata confidence: {avg_conf*100:.1f}%")
        print(f"{'═'*50}")


if __name__ == "__main__":
    main()
