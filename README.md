# Depth Estimation — Offline Evaluation Toolkit

**Offline depth-accuracy evaluation for YOLO26-Depth and Depth Anything V2 — PyTorch and OpenVINO checkpoints, ScanNet benchmarks and your own captured lab data — one consistent metric pipeline across all of it.**

---

## Repo layout

```
depth-estimation-eval-toolkit/
├── colab/
│   └── offline_eval.py              # Colab/Kaggle: ScanNet setup + every model/format eval
└── local/
    ├── offline_depth_eval.py        # library: run_offline_eval() — YOLO26 PyTorch
    ├── offline_depth_eval_dav2.py   # library: run_offline_eval_dav2() — DAv2 PyTorch
    ├── run_lab_eval.py              # standalone script — run against your local lab dataset (YOLO26)
    └── run_lab_eval_openvino.py     # standalone script — run against your local lab dataset (OpenVINO)
```

Two different environments, two different folders:

- **`colab/`** — the ScanNet-benchmark workflow. Downloads the full `scannet_frames_25k` dataset fresh each run, builds the fixed test split, and evaluates YOLO26-Depth, DAv2, and OpenVINO checkpoints against it. Meant to run in Colab/Kaggle, where downloading a multi-GB dataset each time is normal.
- **`local/`** — the lab-generalization workflow. No dataset download — it points at RealSense captures already sitting on your own machine and evaluates a checkpoint against those. Meant to run directly on your dev machine (`python3 run_lab_eval.py`), no cloud environment needed.

Both produce the exact same CSV format and `=== Aggregate ===` summary (`AbsRel`, `RMSE`, `LogRMSE`, `Delta1-3`), so a ScanNet result and a lab result are directly comparable — that consistency is the entire point of keeping this as one toolkit instead of ad hoc scripts per experiment.

---

## How the pieces depend on each other

`run_lab_eval.py` doesn't reimplement the evaluation loop — it **imports** `run_offline_eval` from `offline_depth_eval.py` (same folder) and just supplies the dataset and checkpoint path. That means a fix to the metric math or model-loading logic in `offline_depth_eval.py` automatically applies the next time `run_lab_eval.py` runs — nothing to keep in sync by hand.

`run_lab_eval_openvino.py` is the one exception — it's currently fully self-contained, with its own `OpenVinoDepthModel` class and its own copy of the metric-computation loop, rather than importing from `offline_depth_eval.py`. Not broken, just worth knowing: a future fix to the shared metric math (like the preprocessing correction below) needs to be applied here separately too, since it isn't wired to import anything. Worth consolidating at some point so there's truly one metric implementation instead of two that happen to currently match.

To evaluate a DAv2 checkpoint locally instead of YOLO26, `run_lab_eval.py` needs one edit — swap `from offline_depth_eval import run_offline_eval` for `from offline_depth_eval_dav2 import run_offline_eval_dav2`, and swap the corresponding call and its `encoder=` argument. There's no separate `run_lab_eval_dav2.py` yet; that's a natural next file to add if lab-eval on DAv2 checkpoints becomes a regular thing.

---

## Where your lab image files go

**Not in this repo.** `run_lab_eval.py` and `run_lab_eval_openvino.py` both expect a local folder with `color/` and `depth/` subfolders (matching numbered filenames), pointed at via the `DATASET_DIR` variable in each script's CONFIG section.

The lab datasets are hosted on Hugging Face: **[WasiqSaleem/lab-depth-eval-datasets](https://huggingface.co/datasets/WasiqSaleem/lab-depth-eval-datasets)**, as three zip files:

| File | Size | Use |
|---|---|---|
| `testing_env_spec_real_camera_dataset_combined.zip` | 26.9 MB | Offline evaluation (`run_lab_eval.py` / `run_lab_eval_openvino.py`) |
| `real_camera_dataset_combined.zip` | 232 MB | Training |
| `training_env_spec_real_camera_dataset_combined.zip` | — | Training |

Download and extract the eval one before running either script:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download WasiqSaleem/lab-depth-eval-datasets \
    testing_env_spec_real_camera_dataset_combined.zip \
    --repo-type dataset --local-dir ./lab-datasets
unzip ./lab-datasets/testing_env_spec_real_camera_dataset_combined.zip -d ./lab-datasets
```

Then point `DATASET_DIR` in the CONFIG section of whichever script you're running at the extracted folder, e.g. `./lab-datasets/testing_env_spec_real_camera_dataset_combined`.

## Preprocessing

`local/run_lab_eval_openvino.py` is used for YOLO26-Depth checkpoints only. Its `preprocess()` method (plain 0-255 → 0-1 scaling with BGR→RGB conversion) matches Ultralytics' export convention.

---

## Related repos

- **Fine-tuning**: [depth-estimation-finetuning-yolo26-dav2](#) — trains the PyTorch checkpoints evaluated here
- **Optimization**: [depth-estimation-openvino-quantization](#) — produces the OpenVINO checkpoints evaluated here
- **ROS 2 integration**: [Fine-Tuned-Depth-Estimation-for-ROS-2](#), [Multi-Model-Monocular-Depth-Estimation-for-ROS-2](#)

## Suggested repo name, description, and tags

**Name:** `depth-estimation-eval-toolkit`

**Description:** "Shared offline evaluation toolkit for depth estimation checkpoints — YOLO26-Depth and Depth Anything V2, PyTorch and OpenVINO, ScanNet benchmarks and local lab-generalization tests, one consistent metric pipeline throughout."

**Tags:** `depth-estimation` `model-evaluation` `pytorch` `openvino` `yolo26` `depth-anything-v2` `computer-vision` `evaluation-toolkit` `scannet` `benchmarking`
