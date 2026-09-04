#!/usr/bin/env python3
"""
Standalone offline depth evaluation — same metric math as the ROS
DepthCSVSummary node, but runs directly over (color_path, depth_path)
pairs instead of subscribing to synced ROS topics. Use this for
NYUv2 / ScanNet held-out test sets so results are directly comparable
to the CSV format you're already using.

Usage (inside your Kaggle/Colab notebook, after training finishes):

    from offline_depth_eval import run_offline_eval

    run_offline_eval(
        model_path="/kaggle/working/runs/depth/scannet_batch3/weights/best.pt",
        test_samples=test_samples,          # list of (color_path, depth_path)
        output_csv="/kaggle/working/scannet_test_eval.csv",
        depth_scale=1.0 / 1000.0,           # ScanNet raw depth is mm -> m
        min_depth=0.1,
        max_depth=20.0,
    )
"""

import os
import csv
import numpy as np
from PIL import Image as PILImage


def run_offline_eval(
    model_path,
    test_samples,
    output_csv,
    depth_scale=1.0 / 1000.0,
    min_depth=0.1,
    max_depth=20.0,
    imgsz=640,
):
    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Same header as your ROS DepthCSVSummary node, minus sync_dt_sec
    # (no sync needed offline — pairs are already exact by construction).
    fieldnames = [
        'frame_id', 'color_path', 'gt_mean', 'pred_mean', 'diff_mean',
        'AbsRel', 'RMSE', 'LogRMSE', 'Delta1', 'Delta2', 'Delta3',
        'valid_pixel_frac',
    ]

    rows = []
    n_skipped = 0

    for frame_id, (color_path, depth_path) in enumerate(test_samples):
        try:
            # ---------------- Ground truth ----------------
            gt_raw = PILImage.open(depth_path)
            gt = np.array(gt_raw).astype(np.float32)

            # Match the unit-handling logic from convert()/check_sample_depth():
            # ScanNet raw depth is typically uint16 millimeters.
            if np.array(gt_raw).dtype == np.uint16:
                gt = gt * depth_scale
            # else: assume already in meters, leave as-is

            # ---------------- Prediction ----------------
            results = model(color_path, imgsz=imgsz, verbose=False)
            result = results[0]

            if result.depth is None:
                print(f"[{frame_id}] Model returned no depth output — skipping.")
                n_skipped += 1
                continue

            pred = result.depth.data.cpu().numpy()
            pred = np.squeeze(pred)
            if pred.ndim != 2:
                print(f"[{frame_id}] Unexpected pred shape {pred.shape} — skipping.")
                n_skipped += 1
                continue

            # Resize prediction to GT resolution if they don't match
            # (model may output at its own working resolution).
            if pred.shape != gt.shape:
                pred_img = PILImage.fromarray(pred)
                pred_img = pred_img.resize(
                    (gt.shape[1], gt.shape[0]), resample=PILImage.BILINEAR
                )
                pred = np.array(pred_img)

        except Exception as e:
            print(f"[{frame_id}] Failed to load/infer: {e}")
            n_skipped += 1
            continue

        # ---------------- Same mask + metric math as DepthCSVSummary ----------------
        mask = (gt >= min_depth) & (gt <= max_depth) & \
               (pred >= min_depth) & (pred <= max_depth)

        if np.count_nonzero(mask) == 0:
            print(f"[{frame_id}] No valid pixels after masking — skipping.")
            n_skipped += 1
            continue

        gt_valid = gt[mask]
        pred_valid = pred[mask]
        diff = np.abs(gt_valid - pred_valid)

        abs_rel = np.mean(diff / gt_valid)
        rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

        gt_log = np.log(np.clip(gt_valid, 1e-6, None))
        pred_log = np.log(np.clip(pred_valid, 1e-6, None))
        log_rmse = np.sqrt(np.mean((gt_log - pred_log) ** 2))

        max_ratio = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
        delta1 = np.mean(max_ratio < 1.25)
        delta2 = np.mean(max_ratio < 1.25 ** 2)
        delta3 = np.mean(max_ratio < 1.25 ** 3)

        row = {
            'frame_id': frame_id,
            'color_path': color_path,
            'gt_mean': float(np.mean(gt_valid)),
            'pred_mean': float(np.mean(pred_valid)),
            'diff_mean': float(np.mean(diff)),
            'AbsRel': float(abs_rel),
            'RMSE': float(rmse),
            'LogRMSE': float(log_rmse),
            'Delta1': float(delta1),
            'Delta2': float(delta2),
            'Delta3': float(delta3),
            'valid_pixel_frac': float(np.count_nonzero(mask) / mask.size),
        }
        rows.append(row)

        if frame_id % 50 == 0:
            print(f"Frame {frame_id}/{len(test_samples)} | "
                  f"AbsRel: {abs_rel:.4f} | RMSE: {rmse:.4f} | D1: {delta1:.4f}")

    # ---------------- Write CSV ----------------
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv} ({n_skipped} skipped)")

    # ---------------- Aggregate summary (matches print_summary() in ROS node) ----------------
    if rows:
        keys = ['AbsRel', 'RMSE', 'LogRMSE', 'Delta1', 'Delta2', 'Delta3']
        means = {k: np.mean([r[k] for r in rows]) for k in keys}
        print(f"\n=== Aggregate over {len(rows)} frames ===")
        for k, v in means.items():
            print(f"{k}: {v:.4f}")
        return means
    else:
        print("No frames evaluated — nothing to summarize.")
        return None
