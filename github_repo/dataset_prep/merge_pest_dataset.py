"""
Merge multiple Roboflow-exported pest zips into one unified YOLO dataset.

This script does NOT assume each zip is single-class. It reads each zip's
own data.yaml to find out how many classes it has and what they're called,
then builds one combined, de-duplicated global class list automatically.

Example: weevil.v1i.yolov8 has nc=2, names=['weevil', 'weevil_damage']
         psyllid.v1i.yolov8 has nc=2, names=['psyllid', 'psyllid_damage']
         leafhopper.v1i.yolov8 might have nc=1, names=['Leaf_hopper_damage']

All class names across all zips get merged into one global ordered list
(duplicates merged by exact name match), and every label file gets its
class id remapped from "local id within its own zip" to "global id".
"""

import os
import shutil
import zipfile
import yaml

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Uses the folder this script is run from as the pest dataset directory.
# Since you run "py merge_pest_dataset.py" from inside the pest dataset folder,
# this will automatically point to the right place.
PEST_DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR       = os.path.join(PEST_DATASET_DIR, "pest_dataset_merged")

# List the zip filenames (without .zip) you want to merge.
# Add/remove entries here to match what's actually in your pest dataset folder.
ZIP_NAMES = [
    "leafhopper.v3i.yolov8",
    "psyllid.v2i.yolov8",
    "Scale_insect.v1i.yolov8",
    "Stem_borer.v1i.yolov8",
    "weevil.v1i.yolov8",
]

# Background folder is NOT a zip - it's an already-extracted folder containing
# pure negative samples (no bounding boxes, label files are empty).
# It only has a "train" split (no "valid"). Path is relative to PEST_DATASET_DIR.
BACKGROUND_FOLDER_NAME = "background"
# ──────────────────────────────────────────────────────────────────────────────


def extract_zip(zip_path, extract_to):
    print(f"  Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def read_local_classes(extracted_root):
    """Read this zip's own data.yaml to get its local class names list."""
    yaml_path = os.path.join(extracted_root, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]  # list, index = local class id


def remap_and_copy_split(extracted_root, split, local_to_global, class_prefix, out_dir):
    """
    Copy images + relabeled label files for one split (train/valid) from one
    zip's extracted folder into the merged output directory, remapping each
    label row's class id from local (per-zip) id to global (merged) id.
    """
    src_img_dir = os.path.join(extracted_root, split, "images")
    src_lbl_dir = os.path.join(extracted_root, split, "labels")

    out_img_dir = os.path.join(out_dir, split, "images")
    out_lbl_dir = os.path.join(out_dir, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.isdir(src_img_dir):
        print(f"    [!] No {split}/images found, skipping.")
        return 0

    count = 0
    for fname in os.listdir(src_img_dir):
        name_noext, ext = os.path.splitext(fname)
        new_fname = f"{class_prefix}_{name_noext}{ext}"

        shutil.copy2(
            os.path.join(src_img_dir, fname),
            os.path.join(out_img_dir, new_fname),
        )

        lbl_fname = name_noext + ".txt"
        src_lbl_path = os.path.join(src_lbl_dir, lbl_fname)
        new_lbl_fname = f"{class_prefix}_{name_noext}.txt"
        dst_lbl_path = os.path.join(out_lbl_dir, new_lbl_fname)

        if os.path.isfile(src_lbl_path):
            with open(src_lbl_path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                local_id = int(parts[0])
                global_id = local_to_global[local_id]
                parts[0] = str(global_id)
                new_lines.append(" ".join(parts))
            with open(dst_lbl_path, "w") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        else:
            open(dst_lbl_path, "w").close()

        count += 1

    return count


def copy_background_split(bg_root, out_dir):
    """
    Copy the background folder's images+labels straight into the merged
    train split. No remapping needed since label files are empty (pure
    negative samples, no objects). Background only has a train split.
    """
    src_img_dir = os.path.join(bg_root, "train", "images")
    src_lbl_dir = os.path.join(bg_root, "train", "labels")

    out_img_dir = os.path.join(out_dir, "train", "images")
    out_lbl_dir = os.path.join(out_dir, "train", "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.isdir(src_img_dir):
        print(f"    [!] No background train/images found at {src_img_dir}, skipping.")
        return 0

    count = 0
    for fname in os.listdir(src_img_dir):
        name_noext, ext = os.path.splitext(fname)
        new_fname = f"background_{name_noext}{ext}"
        shutil.copy2(
            os.path.join(src_img_dir, fname),
            os.path.join(out_img_dir, new_fname),
        )

        lbl_fname = name_noext + ".txt"
        src_lbl_path = os.path.join(src_lbl_dir, lbl_fname)
        new_lbl_fname = f"background_{name_noext}.txt"
        dst_lbl_path = os.path.join(out_lbl_dir, new_lbl_fname)

        # background labels should be empty (no objects) - just copy as-is,
        # or create empty file if missing
        if os.path.isfile(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            open(dst_lbl_path, "w").close()

        count += 1

    return count


def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Output dir already exists, removing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temp_extract_dir = os.path.join(OUTPUT_DIR, "_temp_extracted")
    os.makedirs(temp_extract_dir, exist_ok=True)

    # ── Pass 1: extract everything, read each zip's local class list ──
    extracted_paths = {}
    local_class_lists = {}

    for zip_name in ZIP_NAMES:
        zip_path = os.path.join(PEST_DATASET_DIR, zip_name + ".zip")
        if not os.path.isfile(zip_path):
            print(f"[!] Zip not found, skipping: {zip_path}")
            continue

        extract_target = os.path.join(temp_extract_dir, zip_name)
        os.makedirs(extract_target, exist_ok=True)
        extract_zip(zip_path, extract_target)

        local_names = read_local_classes(extract_target)
        extracted_paths[zip_name] = extract_target
        local_class_lists[zip_name] = local_names
        print(f"  -> {zip_name}: local classes = {local_names}")

    # ── Build global class list (merge by exact name match, preserve first-seen order) ──
    global_classes = []
    for zip_name in ZIP_NAMES:
        if zip_name not in local_class_lists:
            continue
        for name in local_class_lists[zip_name]:
            if name not in global_classes:
                global_classes.append(name)

    global_index = {name: i for i, name in enumerate(global_classes)}

    print("\n=== Global merged class list ===")
    for i, name in enumerate(global_classes):
        print(f"  {i}: {name}")

    # ── Pass 2: copy + remap each zip's labels into the merged dataset ──
    summary = {}
    for zip_name in ZIP_NAMES:
        if zip_name not in extracted_paths:
            continue

        print(f"\n=== Merging {zip_name} ===")
        extract_target = extracted_paths[zip_name]
        local_names = local_class_lists[zip_name]

        # map this zip's local id -> global id
        local_to_global = {
            local_id: global_index[name]
            for local_id, name in enumerate(local_names)
        }

        class_prefix = zip_name.split(".")[0].lower()

        train_count = remap_and_copy_split(
            extract_target, "train", local_to_global, class_prefix, OUTPUT_DIR
        )
        valid_count = remap_and_copy_split(
            extract_target, "valid", local_to_global, class_prefix, OUTPUT_DIR
        )

        summary[zip_name] = {"train": train_count, "valid": valid_count}
        print(f"  train: {train_count} images | valid: {valid_count} images")

    shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # ── Copy background (pure negative) images into merged train split ──
    bg_root = os.path.join(PEST_DATASET_DIR, BACKGROUND_FOLDER_NAME)
    bg_count = 0
    if os.path.isdir(bg_root):
        print(f"\n=== Merging background (negative samples) ===")
        bg_count = copy_background_split(bg_root, OUTPUT_DIR)
        print(f"  background: {bg_count} images added to train")
    else:
        print(f"\n[!] Background folder not found at {bg_root}, skipping.")

    # ── Write combined data.yaml ──
    data_yaml = {
        "train": "./train/images",
        "val": "./valid/images",
        "nc": len(global_classes),
        "names": global_classes,
    }
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"Merged dataset saved to: {OUTPUT_DIR}")
    print("\nPer-zip image counts:")
    total_train, total_valid = 0, 0
    for zname, counts in summary.items():
        print(f"  {zname:30s} train={counts['train']:4d}  valid={counts['valid']:4d}")
        total_train += counts["train"]
        total_valid += counts["valid"]
    print(f"  {'TOTAL':30s} train={total_train:4d}  valid={total_valid:4d}")
    print(f"  {'background (train only)':30s} {bg_count:4d}")
    print(f"\nFinal data.yaml has {len(global_classes)} classes:")
    for i, name in enumerate(global_classes):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
