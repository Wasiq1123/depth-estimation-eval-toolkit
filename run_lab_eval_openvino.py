#!/usr/bin/env python3
"""
Standalone LOCAL eval script for OpenVINO (.xml) depth models.

Same metric math as run_lab_eval.py / offline_depth_eval.py (AbsRel, RMSE,
LogRMSE, Delta1-3), but runs inference through OpenVINO instead of the
Ultralytics YOLO() wrapper — mirroring the exact preprocess/extract_depth
logic from your DepthEstimationNode ROS node, just without ROS.

Requirements (install once):
    pip install openvino pillow numpy opencv-python

Usage:
    python3 run_lab_eval_openvino.py
    (edit the CONFIG section below first)
"""

import os
import csv
import time

import cv2
import numpy as np
import openvino as ov
from PIL import Image as PILImage

# ---------------- CONFIG — edit these ----------------

DATASET_DIR = "/home/wasiq/testing_env_spec_real_camera_dataset_combined"  # folder to evaluate on

MODEL_PATH = "/home/wasiq/Downloads/save_models/yolo_depth/office_finetune_nano_v5_best.xml"

OUTPUT_CSV = "/home/wasiq/office_finetune_nano_v5_best.csv"

DEPTH_SCALE = 1.0 / 1000.0   # RealSense/ScanNet: uint16 mm -> meters
MIN_DEPTH = 0.1
MAX_DEPTH = 20.0

# -------------------------------------------------------


def build_lab_samples(dataset_dir):
    color_dir = os.path.join(dataset_dir, "color")
    depth_dir = os.path.join(dataset_dir, "depth")
    samples = []
    for fname in sorted(os.listdir(color_dir)):
        idx = os.path.splitext(fname)[0]
        color_path = os.path.join(color_dir, fname)
        depth_path = os.path.join(depth_dir, f"{idx}.png")
        if os.path.exists(depth_path):
            samples.append((color_path, depth_path))
    return samples


class OpenVinoDepthModel:
    """Wraps the same preprocess / inference / extract_depth logic as
    DepthEstimationNode, minus the ROS plumbing."""

    def __init__(self, model_path):
        self.core = ov.Core()
        print(f"Loading OpenVINO model: {model_path}")
        model = self.core.read_model(model_path)

        self.input_port = model.inputs[0]
        self.input_name = self.input_port.get_any_name()
        input_shape_obj = self.input_port.partial_shape

        if not input_shape_obj.is_static:
            raise RuntimeError(
                "Model input shape is dynamic. This script expects a "
                "static YOLO export, same as DepthEstimationNode."
            )

        input_shape = list(input_shape_obj.to_shape())
        if len(input_shape) != 4:
            raise RuntimeError(f"Expected 4D BCHW input, got {input_shape}")

        self.model_batch = input_shape[0]
        self.model_channels = input_shape[1]
        self.model_height = input_shape[2]
        self.model_width = input_shape[3]

        print(
            f"Model input: batch={self.model_batch}, "
            f"channels={self.model_channels}, "
            f"height={self.model_height}, width={self.model_width}"
        )

        self.compiled_model = self.core.compile_model(model, "AUTO")
        print(f"Running on: {self.compiled_model.get_property('EXECUTION_DEVICES')}")

        # Warmup
        dummy = np.zeros(
            (self.model_batch, self.model_channels, self.model_height, self.model_width),
            dtype=np.float32,
        )
        self.compiled_model({self.input_name: dummy})
        print("Warmup complete.")

    def preprocess(self, rgb):
        resized = cv2.resize(
            rgb, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR
        )
        image = resized.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        if self.model_batch > 1:
            image = np.repeat(image, self.model_batch, axis=0)
        return image

    def extract_depth(self, result):
        depth = result[self.compiled_model.outputs[0]]
        depth = np.asarray(depth)

        if depth.ndim == 4:
            depth = depth[0]
        elif depth.ndim == 3:
            if depth.shape[0] == self.model_batch or depth.shape[0] == 1:
                depth = depth[0]

        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[:, :, 0]

        if depth.ndim != 2:
            raise RuntimeError(f"Could not reduce model output to 2D. Shape: {depth.shape}")

        return depth

    def infer(self, rgb_uint8):
        """rgb_uint8: HxWx3 RGB uint8 array. Returns depth resized back to
        the input image's original resolution."""
        original_h, original_w = rgb_uint8.shape[:2]
        tensor = self.preprocess(rgb_uint8)

        t0 = time.perf_counter()
        result = self.compiled_model({self.input_name: tensor})
        infer_dt = time.perf_counter() - t0

        depth = self.extract_depth(result)
        depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        depth = depth.astype(np.float32)

        return depth, infer_dt


def run_eval(model, test_samples, output_csv, depth_scale, min_depth, max_depth):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        'frame_id', 'color_path', 'gt_mean', 'pred_mean', 'diff_mean',
        'AbsRel', 'RMSE', 'LogRMSE', 'Delta1', 'Delta2', 'Delta3',
        'valid_pixel_frac', 'infer_ms',
    ]

    rows = []
    n_skipped = 0

    for frame_id, (color_path, depth_path) in enumerate(test_samples):
        try:
            gt_raw = PILImage.open(depth_path)
            gt = np.array(gt_raw).astype(np.float32)
            if np.array(gt_raw).dtype == np.uint16:
                gt = gt * depth_scale

            rgb = cv2.imread(color_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            pred, infer_dt = model.infer(rgb)

            if pred.shape != gt.shape:
                pred_img = PILImage.fromarray(pred)
                pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), resample=PILImage.BILINEAR)
                pred = np.array(pred_img)

        except Exception as e:
            print(f"[{frame_id}] Failed to load/infer: {e}")
            n_skipped += 1
            continue

        mask = (gt >= min_depth) & (gt <= max_depth) & (pred >= min_depth) & (pred <= max_depth)
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
            'infer_ms': float(infer_dt * 1000.0),
        }
        rows.append(row)

        if frame_id % 20 == 0:
            print(f"Frame {frame_id}/{len(test_samples)} | AbsRel: {abs_rel:.4f} | "
                  f"D1: {delta1:.4f} | infer: {infer_dt*1000:.1f}ms")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv} ({n_skipped} skipped)")

    if rows:
        keys = ['AbsRel', 'RMSE', 'LogRMSE', 'Delta1', 'Delta2', 'Delta3', 'infer_ms']
        means = {k: np.mean([r[k] for r in rows]) for k in keys}
        print(f"\n=== Aggregate over {len(rows)} frames ===")
        for k, v in means.items():
            print(f"{k}: {v:.4f}")
        return means
    else:
        print("No frames evaluated — nothing to summarize.")
        return None


def main():
    print(f"Building sample list from: {DATASET_DIR}")
    samples = build_lab_samples(DATASET_DIR)
    print(f"Found {len(samples)} pairs.")

    if len(samples) == 0:
        print("No samples found — check DATASET_DIR. Aborting.")
        return

    model = OpenVinoDepthModel(MODEL_PATH)

    run_eval(model, samples, OUTPUT_CSV, DEPTH_SCALE, MIN_DEPTH, MAX_DEPTH)


if __name__ == "__main__":
    main()
