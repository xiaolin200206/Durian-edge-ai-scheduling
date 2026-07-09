"""
Pest Detection Model Training - YOLOv8n vs YOLOv11n Comparison
7 classes: leafhopper damage, Psyllid, Psyllid_damage, Scale_insect,
           Stem-borer, weevil, weevil_damage
Plus background negative samples for improved precision.
"""

from ultralytics import YOLO
import torch
import os
import json
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_YAML   = r"C:\Users\your\dataset\pest dataset\pest_dataset_merged\data.yaml"
SAVE_DIR    = r"C:\Users\your\dataset\pest_experiments"

EPOCHS      = 150
IMGSZ       = 640
BATCH       = 16
PATIENCE    = 30          # a bit higher than leaf model since pest dataset is smaller/more imbalanced
WORKERS     = 4
DEVICE      = 0

MODELS = {
    "YOLOv8n":  "yolov8n.pt",
    "YOLOv11n": "yolo11n.pt",
}
# ──────────────────────────────────────────────────────────────────────────────


def train_and_validate(model_name, weights_path):
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")

    model = YOLO(weights_path)

    start_time = time.time()
    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = SAVE_DIR,
        name        = model_name,
        exist_ok    = True,

        # augmentation - moderately strong given limited samples per class
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.3,
        fliplr      = 0.5,
        mosaic      = 1.0,

        plots       = True,
        save        = True,
        verbose     = True,
    )
    train_time = time.time() - start_time

    # ─── Validation ───────────────────────────────────────────────────────────
    print(f"\nRunning validation for {model_name}...")
    best_weights = os.path.join(SAVE_DIR, model_name, "weights", "best.pt")
    val_model = YOLO(best_weights)
    val_results = val_model.val(
        data    = DATA_YAML,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        split   = "val",
        plots   = True,
    )

    # ─── ONNX export ──────────────────────────────────────────────────────────
    print(f"\nExporting {model_name} to ONNX...")
    val_model.export(format="onnx", imgsz=IMGSZ)

    # ─── Collect per-class results ───────────────────────────────────────────
    class_names = val_results.names
    per_class_ap50 = {}
    if hasattr(val_results.box, "ap50") and val_results.box.ap50 is not None:
        for i, name in class_names.items():
            try:
                per_class_ap50[name] = float(val_results.box.ap50[i])
            except (IndexError, TypeError):
                per_class_ap50[name] = None

    summary = {
        "model": model_name,
        "epochs": EPOCHS,
        "train_time_seconds": round(train_time, 1),
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "per_class_AP50": per_class_ap50,
        "weights_path": best_weights,
    }

    print(f"\n--- {model_name} Results ---")
    print(f"  mAP50:     {summary['mAP50']:.4f}")
    print(f"  mAP50-95:  {summary['mAP50-95']:.4f}")
    print(f"  Precision: {summary['precision']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f}")
    print(f"  Train time: {summary['train_time_seconds']}s")
    print(f"  Per-class AP50:")
    for cname, ap in per_class_ap50.items():
        print(f"    {cname:25s}: {ap:.4f}" if ap is not None else f"    {cname:25s}: N/A")

    return summary


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = {}
    for model_name, weights_path in MODELS.items():
        summary = train_and_validate(model_name, weights_path)
        all_results[model_name] = summary

    # ─── Save comparison JSON ────────────────────────────────────────────────
    comparison_path = os.path.join(SAVE_DIR, "pest_comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    for model_name, summary in all_results.items():
        print(f"\n{model_name}:")
        print(f"  mAP50:     {summary['mAP50']:.4f}")
        print(f"  mAP50-95:  {summary['mAP50-95']:.4f}")
        print(f"  Precision: {summary['precision']:.4f}")
        print(f"  Recall:    {summary['recall']:.4f}")

    print(f"\nFull results saved to: {comparison_path}")
    print("\nNote: Background (negative) images contribute to precision by")
    print("teaching the model what NOT to detect, but don't appear as a")
    print("separate class in per-class AP - they only affect false positive rate.")


if __name__ == "__main__":
    main()
