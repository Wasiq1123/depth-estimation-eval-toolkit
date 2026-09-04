#!/usr/bin/env python3
"""
Standalone offline depth evaluation for Depth Anything V2 — same metric
math and CSV format as offline_depth_eval.py's YOLO26 version (and the
ROS DepthCSVSummary node), so results are directly comparable across
both model families. Runs directly over (color_path, depth_path) pairs.

Usage (inside your Kaggle/Colab notebook, after training finishes):

    from offline_depth_eval_dav2 import run_offline_eval_dav2

    run_offline_eval_dav2(
        model_path="/kaggle/input/models/wasiqsaleem/dav2/pytorch/default/1/depth_anything_v2_metric_hypersim_vits.pth",
        test_samples=test_samples,          # list of (color_path, depth_path)
        output_csv="/kaggle/working/base_dav2_scannet_test_eval.csv",
        encoder="vits",
        depth_scale=1.0 / 1000.0,           # ScanNet raw depth is mm -> m
        min_depth=0.1,
        max_depth=20.0,
    )

Note: this expects the Depth-Anything-V2 repo (the `depth_anything_v2`
package) to be importable — same as in your fine-tuning notebooks, e.g.
sys.path.append('/kaggle/input/.../Depth-Anything-V2') or
sys.path.append('/kaggle/input/.../Depth-Anything-V2/metric_depth')
before importing this module.
"""

import os
import csv
import numpy as np
import torch
import cv2
from PIL import Image as PILImage


# Same architecture configs as Depth-Anything-V2's own scripts
DAV2_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def load_dav2_model(model_path, encoder='vits', max_depth=20.0, device=None):
    """Load a Depth Anything V2 metric-depth model from a checkpoint.

    Handles both a bare state_dict (like the stock Hypersim release
    checkpoints) and a full training-checkpoint dict that also carries
    epoch/optimizer/loss keys (like your fine-tuned checkpoints) — adjust
    the key name below if your training loop saved it under something
    other than 'model_state_dict'.
    """
    from depth_anything_v2.dpt import DepthAnythingV2

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DepthAnythingV2(**{**DAV2_MODEL_CONFIGS[encoder], 'max_depth': max_depth})

    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Already a bare state_dict (e.g. the stock Hypersim release .pth)
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def run_offline_eval_dav2(
    model_path,
    test_samples,
    output_csv,
    encoder='vits',
    depth_scale=1.0 / 1000.0,
    min_depth=0.1,
    max_depth=20.0,
    device=None,
):
    print(f"Loading model: {model_path}")
    model = load_dav2_model(model_path, encoder=encoder, max_depth=max_depth, device=device)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

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
            # DAv2's infer_image takes a raw BGR image (cv2-style) and
            # returns a metric-depth map already resized to the input's
            # own H, W — no imgsz/letterboxing step like the YOLO path.
            raw_img = cv2.imread(color_path)
            if raw_img is None:
                print(f"[{frame_id}] Failed to read image: {color_path} — skipping.")
                n_skipped += 1
                continue

            with torch.no_grad():
                pred = model.infer_image(raw_img)

            pred = np.squeeze(np.asarray(pred))
            if pred.ndim != 2:
                print(f"[{frame_id}] Unexpected pred shape {pred.shape} — skipping.")
                n_skipped += 1
                continue

            # Resize prediction to GT resolution if they don't match
            # (safety net — infer_image usually already matches raw_img's size).
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
