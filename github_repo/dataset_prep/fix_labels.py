"""
Fix corrupt YOLO label files
Problem: '0.02578125\\n2' instead of actual newline
"""

import os
import glob

DATASET_PATH = r"C:\Users\your\leaf_dataset"

def fix_label_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace literal \n string with actual newline
    if '\\n' in content:
        fixed = content.replace('\\n', '\n')
        # Clean up any double newlines
        lines = [l.strip() for l in fixed.splitlines() if l.strip()]
        fixed = '\n'.join(lines) + '\n'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        return True
    return False

def fix_all_labels(dataset_path):
    fixed = 0
    skipped = 0
    errors = 0

    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_dir):
            print(f"Skipping {split} - folder not found")
            continue

        txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
        print(f"\n{split}: found {len(txt_files)} label files")

        for f in txt_files:
            try:
                if fix_label_file(f):
                    fixed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  ERROR: {f} -> {e}")
                errors += 1

    print(f"\n{'='*40}")
    print(f"Fixed:   {fixed} files")
    print(f"Clean:   {skipped} files (no issue)")
    print(f"Errors:  {errors} files")
    print(f"{'='*40}")
    print("Done! Re-run training now.")

if __name__ == "__main__":
    fix_all_labels(DATASET_PATH)
