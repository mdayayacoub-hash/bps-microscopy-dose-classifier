# NASA BPS Microscopy – DNA Damage Dose Classification

This repository contains a feature-based machine learning pipeline for classifying low vs high radiation dose DNA damage in fluorescence microscopy images from the NASA Biological and Physical Sciences (BPS) Microscopy Benchmark Training Dataset.

The project includes:
- A training script that extracts biologically motivated features from microscopy images and trains a neural network classifier.
- A standalone GUI application that loads the trained model, performs inference on new images, and visualizes predictions interactively.
- A manual neural-network forward pass at inference time (no scikit-learn dependency in the GUI).

---

## Scientific Background

The dataset consists of fluorescence microscopy images of individual nuclei from mouse fibroblast cells, irradiated with either high-energy Fe particles or X-rays.

- DNA damage is visualized via 53BP1 foci, a marker of double-strand break repair.
- Images are maximum-intensity projections of 9-slice Z-stacks.
- Radiation dose is measured in Gray (Gy).

This project formulates a binary classification task using only clearly separated dose ranges:

| Class | Dose definition |
|------|-----------------|
| Low dose | ≤ 0.30 Gy |
| High dose | ≥ 0.82 Gy |

Intermediate doses are intentionally excluded to create a clean, high-confidence classification problem.

---

## Repository Structure

bps-microscopy-dose-classifier/
├── train.py            # Feature extraction + MLP training + export
├── gui.py              # Tkinter GUI for inference and visualization
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Ignore data and generated artifacts

---

## Dataset

### Official Source

The dataset is publicly hosted by NASA as part of the AWS Open Data Registry:

BPS Microscopy Benchmark Dataset  
https://registry.opendata.aws/bps_microscopy/

- No authentication required
- No usage restrictions
- Optimized for ML benchmarking

### Subset Used in This Project

This project uses the FITC / 53BP1 channel and its accompanying metadata:

s3://nasa-bps-training-data/Microscopy/train/

This directory contains:
- Fluorescence TIFF images (.tif, .tiff)
- meta.csv containing filenames and radiation dose (Gy)

---

## Downloading the Data

### Prerequisite

Install the AWS CLI:  
https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

### Download Command

aws s3 sync --no-sign-request \
  s3://nasa-bps-training-data/Microscopy/train/ \
  ./Microscopy/train/ \
  --region us-west-2

After download, your local directory should look like:

Microscopy/train/
├── meta.csv
├── *.tif
└── ...

---

## Environment Setup

### Python Version

Python 3.9 or newer is recommended.

### Install Dependencies

pip install -r requirements.txt

### requirements.txt Contents

numpy  
pandas  
opencv-python  
scikit-image  
matplotlib  
scikit-learn  
Pillow  

---

## Training the Model

### Configure Paths (if needed)

By default, train.py expects:

META_CSV = "meta.csv"  
IMAGE_ROOT = "train"

You can either:
- Run the script from inside the Microscopy/ directory, or
- Edit these paths to point to your local dataset

### Run Training

python train.py

### Training Pipeline Overview

1. Load and filter metadata to low/high dose samples
2. Match TIFF images to metadata
3. Extract three handcrafted foci-like features per image:
   - Mean top-hat signal
   - Max top-hat signal
   - Number of detected foci (LoG blobs)
4. Standardize features using training data only
5. Train a multi-layer perceptron (MLP) with manual epoch control
6. Apply validation-based early stopping
7. Save training curves and export model parameters

### Training Outputs

nn_artifacts.json  
training_curves.png  

---

## Running the GUI

### Ensure Trained Model Exists

The GUI requires:

nn_artifacts.json

### Launch the GUI

python gui.py

### Using the GUI

- Click Select Folder…
- Choose a folder containing TIFF images
- Navigate images using Previous / Next
- View:
  - Predicted class (Low vs High dose)
  - Probability of high-dose DNA damage
  - Extracted feature values
  - Contrast-adjusted image preview

### UI Conventions

- Green → High dose
- Red → Low dose
- Classification threshold: P(high) ≥ 0.5

---

## Model Architecture

- Input: 3 handcrafted features
- Hidden layers: 128 → 64 → 32 (ReLU activation)
- Output: Binary classification
- Optimizer: Adam
- Loss function: Cross-entropy (log loss)

The GUI performs inference using a manual NumPy forward pass and does not depend on scikit-learn.

---

## Design Philosophy

This project emphasizes:
- Interpretable, biologically motivated features
- Strict preprocessing parity between training and inference
- Explicit export of learned parameters for reproducibility
- Clear separation between model training and deployment

It is intended as:
- A baseline ML benchmark
- A microscopy-focused ML demonstration
- A teaching and visualization tool

---

## Notes and Limitations

- Only extreme dose ranges are used
- Blob detection parameters are heuristic
- Per-image normalization removes absolute intensity scale
- Duplicate filenames across folders may overwrite during training

---

## License and Data Use

- Code: add a license of your choice (MIT recommended)
- Data: provided by NASA BPS with no restrictions on use

---

## Acknowledgements

- NASA Biological and Physical Sciences Division
- BPS Open Science Program
- NumPy, scikit-learn, OpenCV, scikit-image, and the Python scientific ecosystem
