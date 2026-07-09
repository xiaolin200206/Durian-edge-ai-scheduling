"""
Leaf Disease Model Training: YOLOv8s vs YOLOv11s
Paper: From Bench to Field - Dual-Model ONNX Deployment
"""

from ultralytics import YOLO
import torch
import os
import json
import time
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_YAML   = r"C:\Users\your\leaf_dataset\data.yaml"
SAVE_DIR    = r"C:\Users\your\leaf_experiments"

EPOCHS      = 150
IMGSZ       = 640
BATCH       = 16        # turunkan ke 8 kalau VRAM tak cukup
PATIENCE    = 20        # early stopping
WORKERS     = 4
DEVICE      = 0         # GPU 0, atau 'cpu'

MODELS = {
    "YOLOv8s":  "yolov8s.pt",
    "YOLOv11s": "yolo11s.pt",
}
# ──────────────────────────────────────────────────────────────────────────────


def train_model(model_name, weights):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    project_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(project_dir, exist_ok=True)

    model = YOLO(weights)

    start = time.time()

    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = project_dir,
        name        = "train",
        exist_ok    = True,

        # augmentation
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.3,
        fliplr      = 0.5,
        mosaic      = 1.0,

        # logging
        plots       = True,
        save        = True,
        save_period = 10,
        verbose     = True,
    )

    elapsed = time.time() - start

    # ─── Export to ONNX ───────────────────────────────────────────────────────
    print(f"\n[{model_name}] Exporting to ONNX...")
    best_weights = os.path.join(project_dir, "train", "weights", "best.pt")
    export_model = YOLO(best_weights)
    export_model.export(format="onnx", imgsz=IMGSZ, simplify=True)
    print(f"[{model_name}] ONNX saved.")

    # ─── Validation ───────────────────────────────────────────────────────────
    print(f"\n[{model_name}] Running validation...")
    val_results = export_model.val(
        data    = DATA_YAML,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        split   = "val",
        plots   = True,
        save_json = True,
    )

    # ─── Save summary ─────────────────────────────────────────────────────────
    summary = {
        "model":        model_name,
        "weights":      weights,
        "epochs_run":   results.epoch if hasattr(results, 'epoch') else EPOCHS,
        "train_time_s": round(elapsed, 1),
        "mAP50":        round(float(val_results.box.map50), 4),
        "mAP50_95":     round(float(val_results.box.map),   4),
        "precision":    round(float(val_results.box.mp),    4),
        "recall":       round(float(val_results.box.mr),    4),
        "best_weights": best_weights,
    }

    summary_path = os.path.join(project_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{model_name}] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = {}

    for model_name, weights in MODELS.items():
        try:
            result = train_model(model_name, weights)
            all_results[model_name] = result
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            all_results[model_name] = {"error": str(e)}

    # ─── Final comparison ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Time(s)':>9}")
    print("-" * 60)

    for name, r in all_results.items():
        if "error" not in r:
            print(f"{name:<12} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
                  f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['train_time_s']:>9.0f}")
        else:
            print(f"{name:<12} ERROR: {r['error']}")

    # Save combined results
    combined_path = os.path.join(SAVE_DIR, "comparison_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {combined_path}")


if __name__ == "__main__":
    main()
