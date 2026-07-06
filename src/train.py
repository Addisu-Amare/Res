
#Reproducible, leak‑free K‑fold cross‑validation for segmentation.

import os
import random
import yaml
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from ultralytics import YOLO
from roboflow import Roboflow

# ================================================================
# 1. Reproducibility – set ALL seeds and deterministic flags
# ================================================================
RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# cuDNN determinism (can slow down training – comment out if speed is critical)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: enforce deterministic operations (PyTorch 2.x)
# torch.use_deterministic_algorithms(True)

# ================================================================
# 2. Configuration
# ================================================================
API_KEY = "your roboflow api"
WORKSPACE = "addisuamarezena"
PROJECT = "segmentation-gn75z"
VERSION = 3
DOWNLOAD_FORMAT = "yolov11"

K_FOLDS = 5
EPOCHS = 100
IMG_SIZE = 640
BATCH = 8
DEVICE = 0          # GPU index or "cpu"
INCLUDE_TEST_IN_CV = False

MODELS = [
    "yolov8n-seg.pt",
    # "yolov8s-seg.pt",
    "yolo11n-seg.pt"
    # "yolo11s-seg.pt"
]

RESULTS_CSV = "cv_yolo_seg_results.csv"

# ================================================================
# 3. Download dataset
# ================================================================
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download(DOWNLOAD_FORMAT)
dataset_path = dataset.location

with open(Path(dataset_path) / "data.yaml", "r") as f:
    original_data = yaml.safe_load(f)
CLASS_NAMES = original_data["names"]
NC = original_data["nc"]
print(f"Dataset: {dataset_path}, classes: {CLASS_NAMES}")

# ================================================================
# 4. Gather image/label pairs
# ================================================================
def gather_files(base_path, subset):
    img_dir = Path(base_path) / subset / "images"
    lbl_dir = Path(base_path) / subset / "labels"
    if not img_dir.exists():
        return []
    pairs = []
    for img_file in img_dir.glob("*"):
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            label_file = lbl_dir / (img_file.stem + ".txt")
            if label_file.exists():
                pairs.append((str(img_file), str(label_file)))
    return pairs

all_pairs = []
all_pairs.extend(gather_files(dataset_path, "train"))
all_pairs.extend(gather_files(dataset_path, "valid"))
if INCLUDE_TEST_IN_CV:
    all_pairs.extend(gather_files(dataset_path, "test"))
print(f"Total samples: {len(all_pairs)}")

# Shuffle – seed already set
np.random.shuffle(all_pairs)

# ================================================================
# 5. K‑Fold cross‑validation (leak‑free loop)
# ================================================================
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_results = []

# Create a temporary directory that will be cleaned up even on error
tmp_base = Path(tempfile.mkdtemp(prefix="yolo_cv_"))
print(f"Temporary folder: {tmp_base}")

try:
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_pairs)):
        print(f"\n{'='*20} Fold {fold+1}/{K_FOLDS} {'='*20}")
        fold_dir = tmp_base / f"fold_{fold}"
        fold_dir.mkdir(parents=True)

        # ----- Create symlinked train/val directories -----
        for split, indices in [("train", train_idx), ("val", val_idx)]:
            img_dest = fold_dir / split / "images"
            lbl_dest = fold_dir / split / "labels"
            img_dest.mkdir(parents=True)
            lbl_dest.mkdir(parents=True)

            for idx in indices:
                img_path, lbl_path = all_pairs[idx]
                dest_img = img_dest / Path(img_path).name
                dest_lbl = lbl_dest / Path(lbl_path).name
                if not dest_img.exists():
                    os.symlink(os.path.abspath(img_path), dest_img)
                if not dest_lbl.exists():
                    os.symlink(os.path.abspath(lbl_path), dest_lbl)

        # ----- Write fold data.yaml -----
        fold_yaml = fold_dir / "data.yaml"
        data_conf = {
            "path": str(fold_dir),
            "train": "train/images",
            "val": "val/images",
            "nc": NC,
            "names": CLASS_NAMES
        }
        with open(fold_yaml, "w") as f:
            yaml.dump(data_conf, f)

        # ----- Train & evaluate each model on this fold -----
        for model_name in MODELS:
            print(f"--- Training {model_name} on fold {fold+1} ---")

            # Create a fresh model (pretrained)
            model = YOLO(model_name)

            model.train(
                data=str(fold_yaml),
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE,
                project=str(tmp_base / "runs"),
                name=f"{model_name.replace('.pt','')}_fold{fold}",
                exist_ok=True,
                verbose=False
            )

            # Validate with best weights (automatically loaded)
            metrics = model.val(split="val")

            fold_results.append({
                "model": model_name,
                "fold": fold + 1,
                "box_mAP50-95": metrics.box.map,
                "box_mAP50": metrics.box.map50,
                "mask_mAP50-95": metrics.seg.map,
                "mask_mAP50": metrics.seg.map50
            })
            print(f"Fold {fold+1} {model_name} mask mAP50-95: {metrics.seg.map:.4f}")

            # ---------- LEAK PREVENTION ----------
            # Delete model and clear GPU cache
            del model
            torch.cuda.empty_cache()
            # -------------------------------------

finally:
    # Always remove temporary files, even if an error occurs
    shutil.rmtree(tmp_base, ignore_errors=True)
    print("Temporary files removed.")

# ================================================================
# 6. Aggregate and save results
# ================================================================
df = pd.DataFrame(fold_results)
summary = df.groupby("model").agg(
    mean_box_mAP50_95=("box_mAP50-95", "mean"),
    std_box_mAP50_95=("box_mAP50-95", "std"),
    mean_box_mAP50=("box_mAP50", "mean"),
    std_box_mAP50=("box_mAP50", "std"),
    mean_mask_mAP50_95=("mask_mAP50-95", "mean"),
    std_mask_mAP50_95=("mask_mAP50-95", "std"),
    mean_mask_mAP50=("mask_mAP50", "mean"),
    std_mask_mAP50=("mask_mAP50", "std"),
).reset_index()

print("\n=============== Cross-Validation Summary ===============")
print(summary.to_string(index=False))
summary.to_csv(RESULTS_CSV, index=False)
print(f"Results saved to {RESULTS_CSV}")