"""
NASA BPS Microscopy Demo GUI
---------------------------
Loads a trained MLP model from nn_artifacts.json, extracts simple “foci-like”
features from TIFF images, runs a manual forward pass, and displays predictions.

This file is intentionally self-contained: no sklearn is required at inference time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from skimage.feature import blob_log


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_JSON = "nn_artifacts.json"

# Feature extraction (must match training exactly)
TOPHAT_KERNEL = 15
BLOB_THRESHOLD = 0.03
BLOB_MIN_SIGMA = 1
BLOB_MAX_SIGMA = 4
BLOB_NUM_SIGMA = 4

# UI rules
COLOR_HIGH = "#1a8f2a"  # High dose -> green
COLOR_LOW = "#c21f1f"   # Low dose  -> red

# Classification threshold shown in the UI
PROB_THRESHOLD = 0.5

# Preview sizing
PREVIEW_MAX_W = 760
PREVIEW_MAX_H = 520

# Percentile contrast stretch for display
DISPLAY_P_LO = 1
DISPLAY_P_HI = 99


# =============================================================================
# Image I/O + Feature Extraction
# =============================================================================

def load_gray_01(path: str) -> np.ndarray:
    """
    Load a TIFF as grayscale float32 normalized to [0, 1].

    Notes:
      - Uses IMREAD_UNCHANGED to preserve bit depth (often uint16).
      - Normalizes by per-image max for robustness across exposures.
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


def extract_foci_features(img01: np.ndarray) -> np.ndarray:
    """
    Compute simple foci-like features:
      1) foci_mean  : mean of top-hat enhanced image
      2) foci_max   : max of top-hat enhanced image
      3) foci_count : number of LoG blobs on top-hat enhanced image
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

    return np.array([foci_mean, foci_max, foci_count], dtype=np.float32)


# =============================================================================
# Manual MLP Inference (from exported sklearn weights)
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Stable softmax for 1D or 2D input.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


class ManualMLP:
    """
    Forward-pass-only neural net using exported weights/biases from sklearn's
    MLPClassifier and saved StandardScaler parameters.

    Important:
      - Hidden layers use ReLU.
      - Output layer may be:
          * 1 unit -> sigmoid for binary probability
          * 2 units -> softmax; probability of class 1 is softmax[..., 1]
        (sklearn often stores 2 outputs for binary classification)
    """

    def __init__(self, artifacts: dict):
        self.mean = np.array(artifacts["preprocessing"]["scaler_mean"], dtype=np.float32)
        self.scale = np.array(artifacts["preprocessing"]["scaler_scale"], dtype=np.float32)

        layers = artifacts["network"]["layers"]
        self.W = [np.array(layer["weights"], dtype=np.float32) for layer in layers]
        self.b = [np.array(layer["biases"], dtype=np.float32) for layer in layers]

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def predict_proba_high(self, x: np.ndarray) -> float:
        """
        Returns P(high_dose == 1) as a float.
        """
        a = self.standardize(x).astype(np.float32)

        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            is_last = (i == len(self.W) - 1)
            if not is_last:
                a = relu(a)

        # Handle either 1-output (sigmoid) or 2-output (softmax) final layer
        a = np.ravel(a)
        if a.shape[0] == 1:
            return float(sigmoid(a)[0])
        if a.shape[0] == 2:
            return float(softmax(a)[1])

        raise ValueError(
            f"Unexpected output size {a.shape[0]} in final layer; expected 1 or 2."
        )


# =============================================================================
# GUI
# =============================================================================

@dataclass
class BrowseState:
    folder_path: Optional[str] = None
    image_files: List[str] = None
    current_index: int = -1

    def __post_init__(self):
        if self.image_files is None:
            self.image_files = []


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NASA DNA Damage Demo (Low vs High Dose) by Aya Yacoub")
        self.root.geometry("1100x700")

        self.state = BrowseState()
        self._imgtk: Optional[ImageTk.PhotoImage] = None  # keep reference to avoid GC

        self.model = self._load_model_or_exit()
        self._build_ui()

    # -------------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------------

    def _load_model_or_exit(self) -> ManualMLP:
        if not os.path.exists(ARTIFACTS_JSON):
            messagebox.showerror(
                "Missing file",
                f"Could not find {ARTIFACTS_JSON} in this folder."
            )
            raise SystemExit(1)

        with open(ARTIFACTS_JSON, "r", encoding="utf-8") as f:
            artifacts = json.load(f)

        return ManualMLP(artifacts)

    def _build_ui(self) -> None:
        # Top bar: folder selection + navigation
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        self.btn_folder = tk.Button(
            top, text="Select Folder...", command=self.pick_folder, height=2, width=18
        )
        self.btn_folder.pack(side="left")

        self.btn_prev = tk.Button(
            top, text="Previous", command=self.prev_image, height=2, width=12, state="disabled"
        )
        self.btn_prev.pack(side="left", padx=(10, 0))

        self.btn_next = tk.Button(
            top, text="Next", command=self.next_image, height=2, width=12, state="disabled"
        )
        self.btn_next.pack(side="left", padx=(8, 0))

        self.path_label = tk.Label(top, text="No folder selected", anchor="w")
        self.path_label.pack(side="left", padx=15, fill="x", expand=True)

        # Middle: left panel (image + results) and right panel (features)
        mid = tk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel
        left = tk.Frame(mid, bd=2, relief="groove")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.image_panel = tk.Label(left, bd=0)
        self.image_panel.pack(side="top", expand=True)

        results = tk.Frame(left)
        results.pack(side="bottom", fill="x", pady=10)

        self.pred_label = tk.Label(results, text="Prediction: -", font=("Segoe UI", 18, "bold"))
        self.pred_label.pack(pady=(0, 8))

        self.prob_label = tk.Label(
            results,
            text="Probability of High Dose of Radiation Damage: -",
            font=("Segoe UI", 12, "bold"),
            wraplength=760,
            justify="center",
        )
        self.prob_label.pack()

        self.file_label = tk.Label(results, text="", font=("Segoe UI", 10))
        self.file_label.pack(pady=(8, 0))

        # Right panel
        right = tk.Frame(mid, bd=2, relief="groove", width=320)
        right.pack(side="right", fill="y")

        tk.Label(right, text="Extracted Features", font=("Segoe UI", 14, "bold")).pack(pady=(12, 10))

        self.f1 = tk.Label(right, text="foci_mean: -", font=("Segoe UI", 12))
        self.f1.pack(pady=4, padx=12, anchor="w")

        self.f2 = tk.Label(right, text="foci_max: -", font=("Segoe UI", 12))
        self.f2.pack(pady=4, padx=12, anchor="w")

        self.f3 = tk.Label(right, text="foci_count: -", font=("Segoe UI", 12))
        self.f3.pack(pady=4, padx=12, anchor="w")

        self.counter_label = tk.Label(right, text="", font=("Segoe UI", 10))
        self.counter_label.pack(pady=(20, 10), padx=12, anchor="w")

    # -------------------------------------------------------------------------
    # Folder browsing
    # -------------------------------------------------------------------------

    def pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder containing TIFF images")
        if not folder:
            return

        files = self._collect_tiffs(folder)
        if not files:
            messagebox.showwarning("No TIFFs found", "This folder has no .tif/.tiff files.")
            self._set_empty_folder(folder)
            return

        self.state.folder_path = folder
        self.state.image_files = files
        self.state.current_index = 0

        self.path_label.config(text=folder)
        self._update_nav_buttons()
        self.load_and_infer_current()

    def _collect_tiffs(self, folder: str) -> List[str]:
        """
        Collect TIFFs only from the top-level of the selected folder.
        """
        files: List[str] = []
        for name in os.listdir(folder):
            low = name.lower()
            if low.endswith(".tif") or low.endswith(".tiff"):
                files.append(os.path.join(folder, name))
        files.sort()
        return files

    def _set_empty_folder(self, folder: str) -> None:
        self.state.folder_path = folder
        self.state.image_files = []
        self.state.current_index = -1
        self.path_label.config(text=folder)
        self._update_nav_buttons()

    def next_image(self) -> None:
        if not self.state.image_files:
            return
        if self.state.current_index < len(self.state.image_files) - 1:
            self.state.current_index += 1
            self.load_and_infer_current()

    def prev_image(self) -> None:
        if not self.state.image_files:
            return
        if self.state.current_index > 0:
            self.state.current_index -= 1
            self.load_and_infer_current()

    def _update_nav_buttons(self) -> None:
        files = self.state.image_files
        idx = self.state.current_index

        if not files or idx < 0:
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            return

        self.btn_prev.config(state=("normal" if idx > 0 else "disabled"))
        self.btn_next.config(state=("normal" if idx < len(files) - 1 else "disabled"))

    # -------------------------------------------------------------------------
    # Inference + UI update
    # -------------------------------------------------------------------------

    def load_and_infer_current(self) -> None:
        try:
            path = self.state.image_files[self.state.current_index]
            fname = os.path.basename(path)

            img01 = load_gray_01(path)
            feats = extract_foci_features(img01)
            prob_high = self.model.predict_proba_high(feats)

            self._update_prediction_labels(prob_high)
            self._update_feature_labels(feats)
            self.file_label.config(text=fname)
            self.counter_label.config(
                text=f"Image {self.state.current_index + 1} / {len(self.state.image_files)}"
            )

            self.show_preview(path)
            self._update_nav_buttons()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _update_prediction_labels(self, prob_high: float) -> None:
        is_high = prob_high >= PROB_THRESHOLD
        pred_text = "High dose" if is_high else "Low dose"
        pred_color = COLOR_HIGH if is_high else COLOR_LOW

        self.pred_label.config(text=f"Prediction: {pred_text}", fg=pred_color)
        self.prob_label.config(
            text=f"Probability of High Dose of Radiation Damage: {prob_high:.3f}",
            fg=pred_color,
        )

    def _update_feature_labels(self, feats: np.ndarray) -> None:
        self.f1.config(text=f"foci_mean:  {feats[0]:.6f}")
        self.f2.config(text=f"foci_max:   {feats[1]:.6f}")
        self.f3.config(text=f"foci_count: {feats[2]:.0f}")

    # -------------------------------------------------------------------------
    # Image preview
    # -------------------------------------------------------------------------

    def show_preview(self, path: str) -> None:
        """
        Display TIFF robustly (avoid washed-out previews):
          - load unchanged (often uint16)
          - grayscale conversion if needed
          - percentile contrast stretch (1..99)
          - convert to 8-bit for Tkinter
          - resize to fit panel
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)

        lo = float(np.percentile(img, DISPLAY_P_LO))
        hi = float(np.percentile(img, DISPLAY_P_HI))

        if hi <= lo:
            lo = float(img.min())
            hi = float(img.max())
            if hi <= lo:
                hi = lo + 1.0

        img01 = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        img8 = (img01 * 255.0).astype(np.uint8)

        pil_img = Image.fromarray(img8, mode="L")
        pil_img = self._resize_to_fit(pil_img, PREVIEW_MAX_W, PREVIEW_MAX_H)

        self._imgtk = ImageTk.PhotoImage(pil_img)
        self.image_panel.config(image=self._imgtk)

    @staticmethod
    def _resize_to_fit(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return pil_img.resize(new_size)


# =============================================================================
# Entrypoint
# =============================================================================

def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
