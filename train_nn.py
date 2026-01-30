"""
NASA BPS Microscopy - Training Script
------------------------------------
Reads meta.csv + TIFF images, extracts simple “foci-like” features, trains an
MLPClassifier to separate low vs high dose, plots training curves, and exports:

  - StandardScaler params (mean/scale)
  - Neural network weights + biases
  - Training metadata (splits, best epoch, etc.)

Output: nn_artifacts.json (consumed by the GUI inference script)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import cv2
from skimage.feature import blob_log

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# =============================================================================
# Configuration
# =============================================================================

META_CSV = "meta.csv"
IMAGE_ROOT = "train"  # TIFF root (scanned recursively)
OUT_JSON = "nn_artifacts.json"

SEED = 42

# Dataset split (requested): 75 / 20 / 5
TRAIN_FRAC = 0.75
VAL_FRAC = 0.20
TEST_FRAC = 0.05

# Debug option: run on subset while iterating
N_SAMPLES: Optional[int] = None  # e.g. 5000

# Dose classification task (clean separation)
LOW_DOSE_MAX = 0.30
HIGH_DOSE_MIN = 0.82

# Top-hat + blob settings (must match inference)
TOPHAT_KERNEL = 15
BLOB_THRESHOLD = 0.03
BLOB_MIN_SIGMA = 1
BLOB_MAX_SIGMA = 4
BLOB_NUM_SIGMA = 4

# MLP settings
HIDDEN_LAYER_SIZES = (128, 64, 32)
ALPHA = 1e-4
BATCH_SIZE = 256
LEARNING_RATE_INIT = 1e-3

# Training loop
EPOCHS = 60
PATIENCE = 10
MIN_DELTA = 1e-4
CURVES_PNG = "training_curves.png"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TrainConfig:
    seed: int = SEED
    train_frac: float = TRAIN_FRAC
    val_frac: float = VAL_FRAC
    test_frac: float = TEST_FRAC
    n_samples: Optional[int] = N_SAMPLES


# =============================================================================
# File discovery + image preprocessing
# =============================================================================

def build_file_map(root_dir: str) -> Dict[str, str]:
    """
    Map filename -> full path for all .tif/.tiff files under root_dir (recursive).

    Note: if duplicate filenames exist in different subfolders, later ones overwrite
    earlier ones in this map.
    """
    file_map: Dict[str, str] = {}
    for root, _, files in os.walk(root_dir):
        for name in files:
            low = name.lower()
            if low.endswith(".tif") or low.endswith(".tiff"):
                file_map[name] = os.path.join(root, name)
    return file_map


def load_gray_01(path: str) -> np.ndarray:
    """
    Load TIFF as grayscale float32 normalized to [0, 1].

    Uses IMREAD_UNCHANGED to preserve original bit depth (common for microscopy).
    Normalization is per-image by max intensity.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    maxv = float(img.max())
    if maxv > 0:
        img /= maxv

    return img


# =============================================================================
# Feature extraction
# =============================================================================

def extract_foci_features(img01: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple ‘foci-like’ features computed on a top-hat enhanced image:

      - foci_mean  : mean intensity of top-hat image
      - foci_max   : max intensity of top-hat image
      - foci_count : blob count (LoG) on top-hat image
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL)
    )
    opened = cv2.morphologyEx(img01, cv2.MORPH_OPEN, kernel)
    tophat = img01 - opened

    foci_mean = float(tophat.mean())
    foci_max = float(tophat.max())

    blobs = blob_log(
        tophat,
        min_sigma=BLOB_MIN_SIGMA,
        max_sigma=BLOB_MAX_SIGMA,
        num_sigma=BLOB_NUM_SIGMA,
        threshold=BLOB_THRESHOLD,
    )
    foci_count = float(len(blobs))

    return foci_mean, foci_max, foci_count


# =============================================================================
# Sampling + splitting
# =============================================================================

def stratified_sample(df: pd.DataFrame, n: Optional[int], label_col: str) -> pd.DataFrame:
    """
    Stratified sample by label_col to preserve class balance.

    If n is None or n >= len(df), returns a full copy.
    """
    if n is None or n >= len(df):
        return df.copy()

    groups: List[pd.DataFrame] = []
    for label, g in df.groupby(label_col):
        frac = len(g) / len(df)
        take = max(1, int(round(frac * n)))
        groups.append(g.sample(n=min(take, len(g)), random_state=SEED))

    return (
        pd.concat(groups)
        .sample(frac=1.0, random_state=SEED)
        .head(n)
        .copy()
    )


def split_75_20_5(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into Train/Val/Test = 75/20/5.

    Implemented as:
      1) take TEST_FRAC from all data
      2) from remaining, take VAL_FRAC / (1 - TEST_FRAC) as validation
    """
    X_rem, X_test, y_rem, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_frac,
        random_state=cfg.seed,
        stratify=y,
    )

    val_from_rem = cfg.val_frac / (1.0 - cfg.test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem,
        y_rem,
        test_size=val_from_rem,
        random_state=cfg.seed,
        stratify=y_rem,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# Plotting
# =============================================================================

def plot_training_curves(history: dict, out_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train log loss")
    plt.plot(epochs, history["val_loss"], label="Val log loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =============================================================================
# Training pipeline
# =============================================================================

def load_and_filter_metadata(meta_csv: str) -> pd.DataFrame:
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Could not find {meta_csv} in current folder.")

    print(f"Loading {meta_csv} ...")
    meta = pd.read_csv(meta_csv)

    required = {"filename", "dose_Gy"}
    if not required.issubset(set(meta.columns)):
        raise ValueError(f"meta.csv must contain columns: {required}. Found: {list(meta.columns)}")

    # Keep only "clean" low/high doses
    meta = meta[(meta["dose_Gy"] <= LOW_DOSE_MAX) | (meta["dose_Gy"] >= HIGH_DOSE_MIN)].copy()
    meta["dose_label"] = np.where(meta["dose_Gy"] >= HIGH_DOSE_MIN, 1, 0)  # 1=high, 0=low

    print(f"Rows after dose filter: {len(meta)}")
    return meta


def attach_paths(meta: pd.DataFrame, image_root: str) -> pd.DataFrame:
    print("Indexing image files (recursive) ...")
    file_map = build_file_map(image_root)
    print("Images found on disk:", len(file_map))

    meta = meta.copy()
    meta["path"] = meta["filename"].map(file_map)
    meta = meta[meta["path"].notna()].copy()

    print(f"Rows with files present: {len(meta)}")
    return meta


def extract_dataset(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert metadata rows into (X, y) by loading images and extracting features.
    Skips any rows that error out.
    """
    X_list: List[List[float]] = []
    y_list: List[int] = []

    total = len(meta)
    bad = 0
    print("Extracting features ...")

    for idx, (_, r) in enumerate(meta.iterrows(), start=1):
        try:
            img01 = load_gray_01(r["path"])
            fmean, fmax, fcount = extract_foci_features(img01)
            X_list.append([fmean, fmax, fcount])
            y_list.append(int(r["dose_label"]))
        except Exception:
            bad += 1

        if idx % 500 == 0 or idx == total:
            print(f"  processed {idx}/{total} (bad/skipped: {bad})")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("Feature matrix:", X.shape)
    print("Labels:", np.bincount(y) if len(y) else "none")

    return X, y


def train_mlp_with_curves(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[MLPClassifier, dict, int, float]:
    """
    Train an sklearn MLP epoch-by-epoch using warm_start to record train/val curves.
    Uses early stopping based on validation log-loss.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=1,            # one epoch per fit call
        warm_start=True,       # continue from previous weights
        shuffle=True,
        random_state=SEED,
        early_stopping=False,  # we manage early stopping ourselves
        verbose=False,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    print("\nTraining MLPClassifier (with curves) ...")

    for epoch in range(1, EPOCHS + 1):
        clf.fit(X_train_s, y_train)

        train_proba = clf.predict_proba(X_train_s)
        val_proba = clf.predict_proba(X_val_s)

        tr_loss = log_loss(y_train, train_proba, labels=[0, 1])
        va_loss = log_loss(y_val, val_proba, labels=[0, 1])

        tr_pred = np.argmax(train_proba, axis=1)
        va_pred = np.argmax(val_proba, axis=1)
        tr_acc = accuracy_score(y_train, tr_pred)
        va_acc = accuracy_score(y_val, va_pred)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
            f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f}"
        )

        # Early stopping
        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "coefs_": [w.copy() for w in clf.coefs_],
                "intercepts_": [b.copy() for b in clf.intercepts_],
            }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val_loss: {best_val:.4f})"
                )
                break

    # Restore best weights
    if best_state is not None:
        clf.coefs_ = best_state["coefs_"]
        clf.intercepts_ = best_state["intercepts_"]
        print(f"Restored best model from epoch {best_epoch} (val_loss={best_val:.4f})")

    return clf, history, best_epoch, best_val


def export_artifacts(
    out_json: str,
    scaler: StandardScaler,
    clf: MLPClassifier,
    history: dict,
    best_epoch: int,
    best_val: float,
    n_samples_used: int,
) -> None:
    artifacts = {
        "task": "low_vs_high_dose",
        "dose_definition": {"low_dose_max": LOW_DOSE_MAX, "high_dose_min": HIGH_DOSE_MIN},
        "features": ["foci_mean", "foci_max", "foci_count"],
        "preprocessing": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        "network": {
            "activation": "relu",
            "layers": [],
        },
        "training": {
            "seed": SEED,
            "n_samples_requested": (None if N_SAMPLES is None else int(N_SAMPLES)),
            "n_samples_used": int(n_samples_used),
            "split": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
            "mlp_hidden_layers": list(HIDDEN_LAYER_SIZES),
            "epochs_ran": len(history["train_loss"]),
            "best_epoch": int(best_epoch if best_epoch != -1 else len(history["train_loss"])),
            "best_val_logloss": float(best_val),
            "curves_png": CURVES_PNG,
        },
    }

    for idx, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        artifacts["network"]["layers"].append(
            {"layer_index": idx, "weights": W.tolist(), "biases": b.tolist()}
        )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nSaved neural network artifacts to: {out_json}")


# =============================================================================
# Entrypoint
# =============================================================================

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    cfg = TrainConfig()

    meta = load_and_filter_metadata(META_CSV)
    meta = attach_paths(meta, IMAGE_ROOT)

    meta = stratified_sample(meta, cfg.n_samples, "dose_label")
    print("Training sample size:", len(meta))
    print("Class balance:\n", meta["dose_label"].value_counts())

    X, y = extract_dataset(meta)
    if len(X) < 200:
        raise RuntimeError("Too few usable samples. Increase N_SAMPLES or check your image paths.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_75_20_5(X, y, cfg)
    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf, history, best_epoch, best_val = train_mlp_with_curves(
        X_train_s, y_train, X_val_s, y_val
    )

    plot_training_curves(history, CURVES_PNG)
    print(f"Saved training curves to: {CURVES_PNG}")

    print("\nValidation performance (final/best model):")
    val_pred = clf.predict(X_val_s)
    print(confusion_matrix(y_val, val_pred))
    print(classification_report(y_val, val_pred, target_names=["Low dose", "High dose"]))

    print("\nTest performance (final/best model):")
    test_pred = clf.predict(X_test_s)
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=["Low dose", "High dose"]))

    export_artifacts(
        OUT_JSON,
        scaler,
        clf,
        history,
        best_epoch,
        best_val,
        n_samples_used=len(X),
    )

    print("Next: run the GUI script (it loads nn_artifacts.json).")


if __name__ == "__main__":
    main()
