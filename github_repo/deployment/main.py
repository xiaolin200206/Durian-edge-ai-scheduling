#!/usr/bin/env python3
# =========================================================
# Durian AI – DUAL MODEL (Leaf + Pest) ONNX CPU
# Optimized: staggered inference timing
# =========================================================
import cv2
import time
import sys
import threading
import numpy as np
import psutil
import csv
import os
import gc
import traceback
import requests
import urllib3
import subprocess
from datetime import datetime
from ultralytics import YOLO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False

HAS_DISPLAY = bool(os.environ.get('DISPLAY'))

# ================= CONFIG =================
LEAF_MODEL_PATH    = "/path/to/models/yolov11s_leaf.onnx"
PEST_MODEL_PATH    = "/path/to/models/yolov11n_pest.onnx"

CONF_THRESH        = 0.35
INFERENCE_SIZE     = 640
MAX_BOX_AREA_RATIO = 0.5
MAX_TEMP_LIMIT     = 82.0

CYCLE_ACTIVE_SEC   = 180
CYCLE_SLEEP_SEC    = 45
LOG_INTERVAL       = 0.5

# ─── Ablation: inference scheduling mode ───────────────────────────────────
# "staggered"  -> pest worker starts 0.4s after leaf (original design)
# "parallel"   -> both workers start at (almost) the same time
# "sequential" -> pest worker waits for leaf to fully finish each cycle
#                 before starting its own inference (see pest_worker logic)
SCHEDULE_MODE = "sequential"   # <-- change this to "staggered" / "sequential" / "parallel"

STAGGER_DELAY_SEC = 0.4   # only used when SCHEDULE_MODE == "staggered"

# Inference intervals (seconds)
LEAF_INTERVAL      = 0.8   # leaf updates every ~0.8s
PEST_INTERVAL      = 1.2   # pest updates every ~1.2s

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"   # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"     # Get via @userinfobot or Telegram API
TELEGRAM_COOLDOWN  = 30.0

SCREEN_W, SCREEN_H = 1024, 600
CSV_FILENAME = f"dual_{SCHEDULE_MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ================= CLASS MAPS =================
# Updated to match the final trained leaf disease model (5 classes):
# algal, leaf_rot, Phomopsis, pink, root
LEAF_MERGE_MAP = {
    "algal":     "Algal Leaf Spot",
    "leaf_rot":  "Leaf Rot",
    "phomopsis": "Phomopsis",
    "pink":      "Pink Disease",
    "root":      "Root Disease",
}

PEST_COLORS = {
    "leafhopper damage": (0,   165, 255),
    "Psyllid":           (255, 255, 0  ),
    "Psyllid_damage":    (255, 200, 0  ),
    "Scale_insect":      (255, 128, 0  ),
    "Stem-borer":        (128, 0,   255),
    "weevil":            (0,   255, 128),
    "weevil_damage":     (0,   200, 100),
}

LEAF_COLOR    = (0, 255, 0)
DEFAULT_COLOR = (200, 200, 200)

# ================= GLOBALS =================
current_frame   = None
leaf_detections = []
pest_detections = []
lock            = threading.Lock()
running         = True

perf_data = {
    "fps": 0.0, "leaf_lat": 0.0, "pest_lat": 0.0,
    "cpu": 0.0, "ram": 0.0, "temp": 0.0, "freq": 0.0
}

# ================= HELPERS =================
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except:
        return 0.0

def get_cpu_freq():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0

# ================= STARTUP CHECK =================
def startup_check():
    passed = True
    print()
    print("=" * 55)
    print("  Durian AI – Dual ONNX  |  Startup Self-Check")
    print("=" * 55)

    for label, path in [("Leaf ONNX", LEAF_MODEL_PATH),
                         ("Pest ONNX", PEST_MODEL_PATH)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  [OK] {label:<20} {size_mb:.1f} MB")
        else:
            print(f"  [FAIL] {label} NOT FOUND: {path}")
            passed = False

    if USE_PICAMERA:
        try:
            tc = Picamera2()
            tc.configure(tc.create_preview_configuration(
                main={"size":(640,640),"format":"RGB888"}))
            tc.start(); time.sleep(0.5)
            f = tc.capture_array(); tc.stop(); tc.close()
            print(f"  [OK] Picamera2               {f.shape}")
        except Exception as e:
            print(f"  [FAIL] Picamera2: {e}"); passed = False
    else:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, f = cap.read(); cap.release()
            if ret: print(f"  [OK] USB Camera              {f.shape}")
            else: print(f"  [FAIL] No frame"); passed = False
        else:
            print(f"  [FAIL] No camera"); passed = False

    temp = get_cpu_temp()
    ram  = psutil.virtual_memory()
    cpu  = psutil.cpu_percent(interval=1)
    tw = " ⚠️ HIGH" if temp > 70 else ""
    print(f"  [OK] CPU {cpu:.0f}%  RAM {ram.available/1024**2:.0f}MB  Temp {temp:.1f}C{tw}")

    try:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hailo_only_folder")
        os.makedirs(test_dir, exist_ok=True)
        tf = os.path.join(test_dir, ".test")
        open(tf, 'w').write("ok"); os.remove(tf)
        print(f"  [OK] Log directory writable")
    except Exception as e:
        print(f"  [FAIL] Log dir: {e}"); passed = False

    if HAS_DISPLAY:
        print(f"  [OK] Display: {os.environ.get('DISPLAY')}")
    else:
        print(f"  [--] Headless — Telegram active")

    print("-" * 55)
    if passed:
        print("  ✅  All checks passed — starting in 3 seconds...")
        print("=" * 55)
        time.sleep(3)
    else:
        print("  ❌  Checks FAILED")
        print("=" * 55)
        sys.exit(1)
    print()

# ================= LOAD MODELS =================
startup_check()

print("Loading Leaf model (yolov11s_leaf.onnx)...")
leaf_model = YOLO(LEAF_MODEL_PATH, task='detect')
leaf_model(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8), verbose=False)
print(f"  Leaf classes: {leaf_model.names}")

print("Loading Pest model (yolov11n_pest.onnx)...")
pest_model = YOLO(PEST_MODEL_PATH, task='detect')
pest_model(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8), verbose=False)
print(f"  Pest classes: {pest_model.names}")
print("Both models loaded.\n")

# ================= INFER =================
def run_leaf_inference(frame_rgb):
    t0 = time.time()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    results = leaf_model(frame_bgr, imgsz=INFERENCE_SIZE, conf=CONF_THRESH,
                         iou=0.45, agnostic_nms=True, verbose=False)
    latency = (time.time() - t0) * 1000
    frame_area = INFERENCE_SIZE * INFERENCE_SIZE
    dets = []
    if results[0].boxes:
        for box in results[0].boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_name = leaf_model.names[cls_id].lower()
            disease  = LEAF_MERGE_MAP.get(raw_name, leaf_model.names[cls_id])
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = bbox
            if (x2-x1)*(y2-y1) > frame_area * MAX_BOX_AREA_RATIO:
                continue
            dets.append({"type":"leaf","disease":disease,"conf":conf,"bbox":bbox})
    return dets, latency

def run_pest_inference(frame_rgb):
    t0 = time.time()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    results = pest_model(frame_bgr, imgsz=INFERENCE_SIZE, conf=CONF_THRESH,
                         iou=0.45, agnostic_nms=True, verbose=False)
    latency = (time.time() - t0) * 1000
    frame_area = INFERENCE_SIZE * INFERENCE_SIZE
    dets = []
    if results[0].boxes:
        for box in results[0].boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            name   = pest_model.names[cls_id]
            bbox   = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = bbox
            if (x2-x1)*(y2-y1) > frame_area * MAX_BOX_AREA_RATIO:
                continue
            dets.append({"type":"pest","disease":name,"conf":conf,"bbox":bbox})
    return dets, latency

# ================= THREADS =================
def sequential_worker():
    """
    True sequential mode: runs leaf inference to completion, THEN runs pest
    inference to completion, in a single thread - one strict cycle at a time.
    This is fundamentally different from staggered/parallel (which use two
    independent threads); sequential guarantees zero temporal overlap between
    the two models' CPU usage.
    """
    global current_frame, leaf_detections, pest_detections, perf_data
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()

        # Step 1: leaf inference (full completion before moving on)
        try:
            leaf_dets, leaf_lat = run_leaf_inference(frame)
        except Exception:
            traceback.print_exc()
            leaf_dets, leaf_lat = [], 0.0
        perf_data["leaf_lat"] = leaf_lat
        with lock:
            leaf_detections = leaf_dets

        # Step 2: pest inference (only starts after leaf is fully done)
        try:
            pest_dets, pest_lat = run_pest_inference(frame)
        except Exception:
            traceback.print_exc()
            pest_dets, pest_lat = [], 0.0
        perf_data["pest_lat"] = pest_lat
        with lock:
            pest_detections = pest_dets

        # Use the longer of the two intervals as the cycle pause, since
        # both models already ran back-to-back in this single cycle
        time.sleep(max(LEAF_INTERVAL, PEST_INTERVAL))


def leaf_worker():
    global current_frame, leaf_detections, perf_data
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()
        try:
            dets, lat = run_leaf_inference(frame)
        except Exception:
            traceback.print_exc()
            dets, lat = [], 0.0
        perf_data["leaf_lat"] = lat
        with lock:
            leaf_detections = dets
        time.sleep(LEAF_INTERVAL)

def pest_worker():
    global current_frame, pest_detections, perf_data
    # Ablation: scheduling mode controls how pest_worker starts relative to leaf_worker
    if SCHEDULE_MODE == "staggered":
        time.sleep(STAGGER_DELAY_SEC)
    elif SCHEDULE_MODE == "parallel":
        pass  # no delay - starts at (almost) the same time as leaf_worker
    # "sequential" mode is handled differently - see note below; this thread
    # still starts immediately but the actual leaf->pest ordering for
    # sequential mode should be implemented as a single combined worker
    # instead of two independent threads (left as-is here since you're
    # running parallel mode first).
    while running:
        if current_frame is None:
            time.sleep(0.1); continue
        frame = current_frame.copy()
        try:
            dets, lat = run_pest_inference(frame)
        except Exception:
            traceback.print_exc()
            dets, lat = [], 0.0
        perf_data["pest_lat"] = lat
        with lock:
            pest_detections = dets
        time.sleep(PEST_INTERVAL)

def monitor_worker():
    global perf_data
    with open(CSV_FILENAME, 'w', newline='') as f:
        csv.writer(f).writerow([
            "Timestamp","FPS","Leaf_Lat_ms","Pest_Lat_ms",
            "CPU_%","RAM_MB","Temp_C","Freq_MHz",
            "Leaf_Detections","Pest_Detections"
        ])
    print(f"Logging: {CSV_FILENAME}")
    while running:
        perf_data["cpu"]  = psutil.cpu_percent(interval=None)
        perf_data["ram"]  = psutil.virtual_memory().used / 1024 / 1024
        perf_data["temp"] = get_cpu_temp()
        perf_data["freq"] = get_cpu_freq()
        with lock:
            l_dets = [d["disease"] for d in leaf_detections]
            p_dets = [d["disease"] for d in pest_detections]
        with open(CSV_FILENAME, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().strftime('%H:%M:%S.%f')[:-3],
                f"{perf_data['fps']:.1f}",
                f"{perf_data['leaf_lat']:.1f}",
                f"{perf_data['pest_lat']:.1f}",
                f"{perf_data['cpu']:.1f}",
                f"{perf_data['ram']:.1f}",
                f"{perf_data['temp']:.1f}",
                f"{perf_data['freq']:.0f}",
                "|".join(l_dets) if l_dets else "None",
                "|".join(p_dets) if p_dets else "None",
            ])
        time.sleep(LOG_INTERVAL)

# ================= TELEGRAM =================
def send_telegram(img_path, message):
    def _send():
        url = f"https://149.154.167.220/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        try:
            with open(img_path, 'rb') as photo:
                r = requests.post(url,
                    data={'chat_id': TELEGRAM_CHAT_ID, 'caption': message},
                    files={'photo': photo},
                    headers={'Host': 'api.telegram.org'},
                    verify=False, timeout=15.0)
            print("✅ Telegram sent!" if r.status_code==200 else f"❌ HTTP {r.status_code}")
        except Exception as e:
            print(f"❌ Telegram: {e}")
    threading.Thread(target=_send, daemon=True).start()

# ================= DASHBOARD =================
def draw_dashboard(img_bgr, leaf_dets, pest_dets):
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (10,10), (340,225), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, f"FPS: {perf_data['fps']:.1f}",
                (20,40), font, 0.7, (0,255,0), 2)
    cv2.putText(img_bgr, f"Leaf:{perf_data['leaf_lat']:.0f}ms Pest:{perf_data['pest_lat']:.0f}ms",
                (20,68), font, 0.55, (0,255,255), 1)
    cv2.putText(img_bgr, f"CPU:{perf_data['cpu']:.0f}% RAM:{perf_data['ram']:.0f}MB",
                (20,93), font, 0.55, (255,255,255), 1)
    temp = perf_data['temp']
    tc = (0,0,255) if temp > 80 else (255,255,255)
    cv2.putText(img_bgr, f"Temp:{temp:.1f}C {perf_data['freq']:.0f}MHz",
                (20,118), font, 0.55, tc, 1)
    l_str = "|".join([d["disease"] for d in leaf_dets]) if leaf_dets else "Clear"
    cv2.putText(img_bgr, f"Leaf:{l_str[:32]}",
                (20,150), font, 0.5, (0,255,0), 1)
    p_str = "|".join([d["disease"] for d in pest_dets]) if pest_dets else "Clear"
    cv2.putText(img_bgr, f"Pest:{p_str[:32]}",
                (20,175), font, 0.5, (0,165,255), 1)
    cv2.putText(img_bgr, "Leaf:YOLOv11s  Pest:YOLOv11n (ONNX CPU)",
                (20,215), font, 0.45, (180,180,180), 1)

# ================= MAIN =================
def main():
    global current_frame, running

    picam2 = cap = None
    if USE_PICAMERA:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size":(640,640),"format":"RGB888"}))
        picam2.start()
        try:
            picam2.set_controls({"AfMode":2,"AfSpeed":1})
            print("Autofocus enabled")
        except:
            pass
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3,1280); cap.set(4,720)

    WIN_NAME = "Durian AI"
    if HAS_DISPLAY:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if SCHEDULE_MODE == "sequential":
        worker_threads = [
            threading.Thread(target=sequential_worker, daemon=True),
            threading.Thread(target=monitor_worker,     daemon=True),
        ]
        print(f"  [Ablation mode: SEQUENTIAL - leaf and pest run back-to-back in one thread]")
    else:
        worker_threads = [
            threading.Thread(target=leaf_worker,    daemon=True),
            threading.Thread(target=pest_worker,    daemon=True),
            threading.Thread(target=monitor_worker, daemon=True),
        ]
        print(f"  [Ablation mode: {SCHEDULE_MODE.upper()}]")

    for t in worker_threads:
        t.start()

    print(f"🟢 Running (CONF={CONF_THRESH}, Leaf every {LEAF_INTERVAL}s, Pest every {PEST_INTERVAL}s)")

    cycle_start        = time.time()
    is_active          = True
    fps_start          = time.time()
    fps_cnt            = 0
    frame_cnt          = 0
    last_telegram_time = 0.0

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hailo_only_folder", f"run_{ts}", "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    try:
        while True:
            now     = time.time()
            elapsed = now - cycle_start

            if is_active:
                if elapsed > CYCLE_ACTIVE_SEC:
                    print("Sleep..."); is_active = False; cycle_start = now; continue
            else:
                if elapsed > CYCLE_SLEEP_SEC:
                    print("Active."); is_active = True; cycle_start = now; fps_start = time.time()
                else:
                    time.sleep(0.5); continue

            temp = get_cpu_temp()
            if temp > MAX_TEMP_LIMIT:
                print(f"OVERHEAT {temp:.1f}C"); time.sleep(5); continue

            if frame_cnt % 100 == 0:
                gc.collect()

            if USE_PICAMERA:
                try:
                    frame_rgb = picam2.capture_array()
                except:
                    continue
            else:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            current_frame = frame_rgb

            fps_cnt += 1
            if fps_cnt >= 10:
                perf_data["fps"] = 10 / (time.time() - fps_start)
                fps_start = time.time(); fps_cnt = 0

            with lock:
                l_dets = list(leaf_detections)
                p_dets = list(pest_detections)

            all_dets = l_dets + p_dets

            vis_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            for d in l_dets:
                x1,y1,x2,y2 = d['bbox']
                cv2.rectangle(vis_bgr,(x1,y1),(x2,y2),LEAF_COLOR,2)
                cv2.putText(vis_bgr,f"{d['disease']} {d['conf']:.2f}",
                            (x1,max(y1-8,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,LEAF_COLOR,2)

            for d in p_dets:
                x1,y1,x2,y2 = d['bbox']
                color = PEST_COLORS.get(d['disease'],DEFAULT_COLOR)
                cv2.rectangle(vis_bgr,(x1,y1),(x2,y2),color,2)
                cv2.putText(vis_bgr,f"{d['disease']} {d['conf']:.2f}",
                            (x1,max(y1-8,12)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)

            draw_dashboard(vis_bgr, l_dets, p_dets)

            if HAS_DISPLAY:
                final = cv2.resize(vis_bgr,(SCREEN_W,SCREEN_H))
                cv2.imshow(WIN_NAME, final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if all_dets and (now - last_telegram_time > TELEGRAM_COOLDOWN):
                img_path = os.path.join(save_dir, f"det_{frame_cnt:06d}.jpg")
                cv2.imwrite(img_path, vis_bgr)
                lines = [f"🍃 {d['disease']}: {d['conf']*100:.1f}%" for d in l_dets]
                lines += [f"🐛 {d['disease']}: {d['conf']*100:.1f}%" for d in p_dets]
                msg = "🚨 Durian Alert!\n\n" + "\n".join(lines) + f"\n\n🌡️{temp:.1f}°C ⏱️{perf_data['fps']:.1f}FPS"
                send_telegram(img_path, msg)
                last_telegram_time = now

            frame_cnt += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception:
        traceback.print_exc()
    finally:
        running = False
        if HAS_DISPLAY:
            cv2.destroyAllWindows()
        try:
            if USE_PICAMERA: picam2.stop()
            elif cap: cap.release()
        except:
            pass
        print(f"CSV: {CSV_FILENAME}")

if __name__ == "__main__":
    main()
