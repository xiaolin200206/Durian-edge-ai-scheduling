"""
Confidence Threshold Sensitivity Analysis
Tests the leaf disease model and pest detection model at multiple confidence
thresholds to justify the CONF_THRESH=0.35 design choice with empirical data,
rather than an arbitrary selection.

Produces a precision-recall trade-off table for each model across thresholds:
0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
"""

from ultralytics import YOLO
import json

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LEAF_MODEL_PATH = r"C:\Users\your\dataset\leaf_experiments\YOLOv11s\train\weights\best.pt"
PEST_MODEL_PATH = r"C:\Users\your\dataset\pest_experiments\YOLOv11n\weights\best.pt"

LEAF_DATA_YAML = r"C:\Users\your\leaf_dataset\data.yaml"
PEST_DATA_YAML = r"C:\Users\your\dataset\pest dataset\pest_dataset_merged\data.yaml"

CONF_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
IMGSZ = 640
BATCH = 16
DEVICE = 0

OUTPUT_JSON = r"C:\Users\your\dataset\confidence_sensitivity_results.json"
# ──────────────────────────────────────────────────────────────────────────────


def run_sensitivity_sweep(model_path, data_yaml, model_label):
    print(f"\n{'='*70}")
    print(f"  Confidence Threshold Sweep: {model_label}")
    print(f"{'='*70}")

    model = YOLO(model_path)
    results_by_threshold = {}

    for conf in CONF_THRESHOLDS:
        print(f"\n--- conf={conf} ---")
        val_results = model.val(
            data=data_yaml,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            conf=conf,
            split="val",
            plots=False,
            verbose=False,
        )

        precision = float(val_results.box.mp)
        recall    = float(val_results.box.mr)
        map50     = float(val_results.box.map50)
        map5095   = float(val_results.box.map)

        # F1 score - useful single metric to identify the "sweet spot" threshold
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  "
              f"F1: {f1:.4f}  mAP50: {map50:.4f}  mAP50-95: {map5095:.4f}")

        results_by_threshold[conf] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
            "mAP50-95": map5095,
        }

    return results_by_threshold


def print_summary_table(results, model_label):
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE: {model_label}")
    print(f"{'='*70}")
    print(f"\n{'Conf':>6s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'mAP50':>8s} {'mAP50-95':>10s}")
    print("-" * 56)
    best_f1_conf = max(results, key=lambda c: results[c]["f1"])
    for conf, r in results.items():
        marker = "  <-- best F1" if conf == best_f1_conf else ""
        print(f"{conf:6.2f} {r['precision']:10.4f} {r['recall']:8.4f} "
              f"{r['f1']:8.4f} {r['mAP50']:8.4f} {r['mAP50-95']:10.4f}{marker}")
    print(f"\nThreshold with best F1 (precision-recall balance): {best_f1_conf}")
    return best_f1_conf


def main():
    all_results = {}

    print("Running sensitivity sweep for LEAF DISEASE model...")
    leaf_results = run_sensitivity_sweep(LEAF_MODEL_PATH, LEAF_DATA_YAML, "Leaf Disease (YOLOv11s)")
    best_leaf_conf = print_summary_table(leaf_results, "Leaf Disease (YOLOv11s)")
    all_results["leaf_disease"] = leaf_results

    print("\n\nRunning sensitivity sweep for PEST DETECTION model...")
    pest_results = run_sensitivity_sweep(PEST_MODEL_PATH, PEST_DATA_YAML, "Pest Detection (YOLOv11n)")
    best_pest_conf = print_summary_table(pest_results, "Pest Detection (YOLOv11n)")
    all_results["pest_detection"] = pest_results

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("  FINAL RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Leaf model best F1 threshold: {best_leaf_conf}")
    print(f"Pest model best F1 threshold: {best_pest_conf}")
    print(f"\nCurrent deployment uses CONF_THRESH=0.35 for both models.")
    print("Compare this against the best-F1 thresholds above to justify (or")
    print("revise) the design choice in the paper's Methods section.")
    print(f"\nFull results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
