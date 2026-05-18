# Anomaly Detection with Convolutional Autoencoder

**Authors:** Pavlos Margaritis & Faris Fejzic

A deep learning pipeline for unsupervised anomaly detection on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), using a convolutional autoencoder trained exclusively on defect-free images. Anomalies are detected at inference time by measuring how poorly the model reconstructs an image.

---

## How It Works

The core idea is simple: train an autoencoder only on **normal (good)** images. Because it has never seen defects, it learns to reconstruct healthy textures and structures well — but struggles to reconstruct anomalous regions. A high reconstruction error therefore signals a defect.

**Reconstruction score** = MSE loss + (1 − SSIM)

At evaluation time, images scoring above a statistically derived threshold are classified as anomalous.

---

## Project Structure

```
├── Code/
│   ├── model.py        # Convolutional autoencoder architecture
│   ├── dataset.py      # MVTec dataset loader with mask support
│   ├── train.py        # Training loop with early stopping
│   ├── threshold.py    # Threshold calibration on good images
│   └── evaluate.py     # Evaluation, metrics, and confusion matrix
├── Datat/              # Dataset directory (not tracked in git)
├── requirements.txt
├── SETUP.txt
└── README.md
```

---

## Model Architecture

The autoencoder uses a symmetric encoder–decoder design operating on 224×224 RGB images.

| Stage       | Layer                        | Output Shape       |
|-------------|------------------------------|--------------------|
| **Encoder** | Conv2d(3 → 64, stride 2)     | 64 × 112 × 112     |
|             | Conv2d(64 → 128, stride 2)   | 128 × 56 × 56      |
|             | Conv2d(128 → 256, stride 2)  | 256 × 28 × 28      |
|             | Conv2d(256 → 512, stride 2)  | 512 × 14 × 14      |
| **Decoder** | ConvTranspose2d(512 → 256)   | 256 × 28 × 28      |
|             | ConvTranspose2d(256 → 128)   | 128 × 56 × 56      |
|             | ConvTranspose2d(128 → 64)    | 64 × 112 × 112     |
|             | ConvTranspose2d(64 → 3)      | 3 × 224 × 224      |

All encoder layers use ReLU activations; the final decoder layer uses Sigmoid to output values in [0, 1].

---

## Setup

### 1. Dataset

- Create a folder named `Datat/` in the project root.
- Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).
- Extract categories (e.g., `bottle`, `cable`) into `Datat/MVTecAD/`.

Expected structure:
```
Datat/MVTecAD/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── broken_large/
    └── ground_truth/
        └── broken_large/
```

### 2. Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python Code/train.py
```

Trains the autoencoder on the `bottle` category for up to 300 epochs with early stopping (patience = 10). The best model is saved as `autoencoder_bottle.pth` and the loss history as `loss_history_bottle.npy`.

> **Note:** A pre-trained `autoencoder_bottle.pth` is already included in the repository — you can skip this step.

### Evaluation

```bash
python Code/evaluate.py
```

This will:
1. Load the saved model weights.
2. Calibrate the anomaly threshold from the training set's good images (mean + 2.7 × std).
3. Run inference on the test set and compute predictions.
4. Display a training loss curve and a confusion matrix.
5. Print classification metrics.

---

## Training Details

| Hyperparameter    | Value              |
|-------------------|--------------------|
| Image size        | 224 × 224          |
| Batch size        | 16                 |
| Epochs (max)      | 300                |
| Learning rate     | 1e-3               |
| Weight decay      | 1e-4               |
| Optimizer         | Adam               |
| Loss function     | MSE + (1 − SSIM)   |
| Early stopping    | Patience = 10      |

---

## Threshold Calibration

Rather than a manually tuned threshold, `threshold.py` derives one statistically from the training set:

```
threshold = mean(good_scores) + 2.7 × std(good_scores)
```

This ensures the threshold adapts to the model's actual reconstruction behaviour for each category.

---

## Evaluation Metrics

After running `evaluate.py`, the following metrics are reported:

- **Accuracy** — overall correct predictions
- **Precision** — of predicted anomalies, how many are truly anomalous
- **Recall** — of all true anomalies, how many were detected
- **F1 Score** — harmonic mean of precision and recall

A confusion matrix plot is also displayed with `Good` and `Anomaly` as class labels.

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

- `torch` / `torchvision`
- `pytorch-msssim`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `numpy`
