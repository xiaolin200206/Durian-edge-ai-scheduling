# Compute–Thermal Co-Design of Co-located Dual-Model Edge Inference — Raspberry Pi 5

Code accompanying the manuscript *"Compute–Thermal Co-Design of Co-located Dual-Model Inference for Agricultural Cyber-Physical Edge Nodes"* (under review, IEEE Transactions on Industrial Cyber-Physical Systems).

This repository contains the on-device deployment/scheduling system, the model-training scripts, and the analysis tools used in the study, together with the raw telemetry logs underlying Table I. Trained model weights and the annotated image datasets are **not** included (see [Data Availability](#data-availability)), but all code and telemetry required to reproduce the scheduling ablation and its statistical analysis are provided.

---

## Repository Structure

```
.
├── deployment/
│   └── main.py                       # Edge deployment script (RPi5 / Jetson) — dual-model
│                                      # ONNX inference with the three scheduling policies,
│                                      # duty-cycle control, Telegram alerting, and CSV
│                                      # telemetry logging
├── training/
│   ├── train_leaf_disease.py         # Leaf-disease model training (YOLOv8s / YOLOv11s)
│   └── train_pest_model.py           # Pest-detection model training (YOLOv8n / YOLOv11n)
├── analysis/
│   ├── analyze_telemetry.py          # Reproduces Table I and every reported
│   │                                  # statistic (dispersion, energy, beat period)
│   │                                  # from the released telemetry logs
│   └── confidence_sensitivity.py     # Confidence-threshold sweep
│                                      # (precision/recall/F1/mAP@0.5 across thresholds)
├── dataset_prep/
│   ├── fix_labels.py                 # Repairs corrupted YOLO label files (literal \n
│   │                                  string artifacts from annotation export)
│   └── merge_pest_dataset.py         # Merges annotation batches + background images
│                                      # into a unified pest-detection dataset
└── data/                             # Raw telemetry logs (three-hour benchmarks; see
                                       # Data Availability). Provided as a released archive.
```

---

## System Overview

The deployment system (`deployment/main.py`) co-locates two YOLO-based ONNX detectors on one passively cooled ARM64 edge node (Raspberry Pi 5, Broadcom BCM2712, four Cortex-A76 cores), executed in FP32 via ONNX Runtime on the CPU:

- **Leaf-disease detector**: YOLOv11s, 5 classes (Algal Leaf Spot, Leaf Rot, Phomopsis, Pink Disease, Root Disease)
- **Pest detector**: YOLOv11n, 7 classes (Leafhopper Damage, Psyllid, Psyllid Damage, Scale Insect, Stem-borer, Weevil, Weevil Damage)

Three inference scheduling policies are implemented and toggled via the `SCHEDULE_MODE` constant:

```python
SCHEDULE_MODE = "sequential"   # "staggered" / "parallel" / "sequential"
```

| Mode | Description |
|---|---|
| `staggered` | Leaf and pest inference run as independent threads with a fixed 0.4 s startup offset |
| `parallel` | Both models run concurrently with no offset (maximum shared-resource contention) |
| `sequential` | A single thread runs leaf inference to completion, then pest inference, so the two never overlap |

The central finding is that scheduling alone — a purely computational choice — moves the node's peak die temperature by **12.7 °C** (81.0 °C under `parallel` vs 68.3 °C under `sequential`) at a statistically indistinguishable capture-loop frame rate, while `sequential` also attains the lowest per-model latency and the tightest latency distribution. See Table I and Sections IV–V of the manuscript for full results.

---

## Setup

### Requirements

```bash
pip install ultralytics onnxruntime opencv-python numpy psutil requests --break-system-packages
```

For GPU-accelerated inference (Jetson platforms):
```bash
pip install onnxruntime-gpu --break-system-packages
```

### Hardware tested
- Raspberry Pi 5 (8 GB), Raspberry Pi Camera Module 3 (IMX708)
- NVIDIA Jetson Orin Nano Developer Kit (cross-platform port)

---

## Reproducing the Experiments

### 1. Dataset preparation
```bash
python dataset_prep/fix_labels.py         # repair corrupted label files before training
python dataset_prep/merge_pest_dataset.py # merge multi-batch pest annotations
```

### 2. Model training
```bash
python training/train_leaf_disease.py   # trains the leaf model (YOLOv8s / YOLOv11s)
python training/train_pest_model.py      # trains the pest model (YOLOv8n / YOLOv11n)
```
Edit the `DATA_YAML` and `SAVE_DIR` paths at the top of each script to point to your local dataset. The YOLOv11 pair is the one deployed in the main scheduling study; the YOLOv8 pair is the earlier model pair used in the accelerator trial of Section VI-B.

### 3. Confidence-threshold sensitivity
```bash
python analysis/confidence_sensitivity.py
```
Sweeps confidence thresholds and reports precision/recall/F1/mAP@0.5 for both models, used to justify the deployed threshold choice.

### 4. Scheduling ablation (on-device)
```bash
# On the edge device (RPi5 or Jetson):
python deployment/main.py
```
Set `SCHEDULE_MODE` at the top of the script before each run (`staggered`, `parallel`, `sequential`). Each policy is run for the same duration (three continuous hours in the manuscript) under consistent ambient conditions. Telemetry is logged every 0.5 s to `dual_<mode>_<timestamp>.csv`.

### 5. Analysis of the telemetry
```bash
python analysis/analyze_telemetry.py --data-dir data
```
This single command recomputes, directly from the released logs, every reported statistic of the scheduling ablation — Table I, the per-inference intervals of Section IV-C, the latency-distribution and beat-period results of Section V, the per-cycle energy of Section IV-D, and the frequency-scaling counts of Section IV-A. (Adjust the `POLICY_FILES` mapping at the top of the script if the released filenames differ.)

The released telemetry logs in `data/` contain every raw sample. Every entry of Table I — per-policy mean ± standard deviation of frame rate, per-model latency, CPU utilization, resident memory, and die temperature — is computed directly from these CSVs, as are the per-inference completion intervals (Section IV-C), the latency-distribution statistics and beat-period episode spacing (Section V), and the TDP-interpolated per-cycle energy (Section IV-D). The frame-rate filter described in Section III-C (samples above 60 FPS and start-up zero-valued samples excluded) has **not** been applied to the released logs; they are provided as recorded.

---

## Data Availability

The raw telemetry logs underlying Table I — three continuous three-hour benchmarks at a 0.5 s sampling interval, comprising 63,952 samples — are released in this repository under `data/` (packaged as an archive). The trained model weights and the annotated durian leaf-disease and pest image datasets are **not** publicly released, as they are proprietary assets of an ongoing commercialization effort. The code in this repository, together with the released telemetry, is sufficient to reproduce the scheduling ablation and every reported statistic on the released logs.

---

## Citation

If you use this code or the released telemetry, please cite:

```
L. D. Shan, "Compute–Thermal Co-Design of Co-located Dual-Model Inference
for Agricultural Cyber-Physical Edge Nodes," submitted to IEEE Transactions
on Industrial Cyber-Physical Systems, 2026.
```

---

## License

Code and telemetry released for research reproducibility. Contact the corresponding author regarding reuse.
