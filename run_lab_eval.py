#!/usr/bin/env python3
"""
Standalone LOCAL eval script — run this on your host machine, no Kaggle needed.

Uses a model checkpoint already on this machine, builds the (color, depth)
pairs from your locally captured RealSense dataset, and runs
offline_depth_eval.py's metric computation over it, writing a CSV in the
same format as your NYUv2/ScanNet comparison tables.

Requirements (install once):
    pip install ultralytics pillow numpy opencv-python

Usage:
    python3 run_lab_eval.py
    (edit the CONFIG section below first)
"""

import os
import sys

# ---------------- CONFIG — edit these ----------------

# Which folder to evaluate on. Use the 65-image folder for the zero-shot
# check; switch to the 35-image folder later for the final held-out test.
DATASET_DIR = "/home/wasiq/testing_env_spec_real_camera_dataset_combined"          # the 65

# Local path to the model checkpoint (.pt) on this machine.
MODEL_PATH = "/home/wasiq/Downloads/save_models/yolo_depth/office_finetune_nano_v5_best.pt"

# Where to write results
OUTPUT_CSV = "/home/wasiq/env_specific_office_finetune_nano_v5_best.csv"

# Where offline_depth_eval.py lives locally (same folder as this script by default)
EVAL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Depth handling — RealSense aligned depth is uint16 millimeters, same as ScanNet.
DEPTH_SCALE = 1.0 / 1000.0
MIN_DEPTH = 0.1
MAX_DEPTH = 20.0
IMGSZ = 640

# -------------------------------------------------------


def build_lab_samples(dataset_dir):
    """Build (color_path, depth_path) pairs from a RealSense capture folder
    that has color/ and depth/ subfolders with matching numbered filenames."""
    color_dir = os.path.join(dataset_dir, "color")
    depth_dir = os.path.join(dataset_dir, "depth")

    if not os.path.isdir(color_dir) or not os.path.isdir(depth_dir):
        raise FileNotFoundError(
            f"Expected {color_dir} and {depth_dir} to both exist. "
            f"Check DATASET_DIR is correct."
        )

    samples = []
    for fname in sorted(os.listdir(color_dir)):
        idx = os.path.splitext(fname)[0]
        color_path = os.path.join(color_dir, fname)
        depth_path = os.path.join(depth_dir, f"{idx}.png")
        if os.path.exists(depth_path):
            samples.append((color_path, depth_path))
        else:
            print(f"Warning: no matching depth file for {fname} — skipping.")

    return samples


def main():
    # ---------------- 1. Build sample list ----------------
    print(f"Building sample list from: {DATASET_DIR}")
    lab_samples = build_lab_samples(DATASET_DIR)
    print(f"Found {len(lab_samples)} color/depth pairs.")

    if len(lab_samples) == 0:
        print("No samples found — check DATASET_DIR path. Aborting.")
        return

    # ---------------- 2. Verify local checkpoint exists ----------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {MODEL_PATH}\n"
            f"Check MODEL_PATH is correct."
        )
    print(f"Using local checkpoint: {MODEL_PATH}")

    # ---------------- 3. Run the eval ----------------
    sys.path.append(EVAL_SCRIPT_DIR)
    from offline_depth_eval import run_offline_eval

    means = run_offline_eval(
        model_path=MODEL_PATH,
        test_samples=lab_samples,
        output_csv=OUTPUT_CSV,
        depth_scale=DEPTH_SCALE,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        imgsz=IMGSZ,
    )

    print("\n=== Done ===")
    print(f"Results written to: {OUTPUT_CSV}")
    if means:
        print("Summary:", means)


if __name__ == "__main__":
    main()