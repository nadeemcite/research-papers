---
title: "U-Net: Code architecture"
parent: "035 U-Net"
nav_order: 1
---
# Code Architecture — U-Net Notebook

## Overview

The notebook builds a U-Net from scratch in PyTorch, trains it on a small synthetic "cells" dataset (64×64 grayscale images of overlapping bright blobs), and compares it with the same network without skip connections. Everything runs in a few minutes on a Kaggle Tesla T4.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, numpy, matplotlib, plus `random`, `os`, `shutil`, `atexit` and `pathlib`.
- Seeds Python, NumPy and PyTorch with 42 and turns on deterministic cuDNN kernels, so a re-run reproduces the numbers in the summary.
- Shows figures as JPEG at 80 dpi so the saved notebook stays small.
- Detects Kaggle from its `KAGGLE_KERNEL_RUN_TYPE` environment variable and sets `OUT_DIR`: the current folder on Kaggle (its output folder), `./outputs` anywhere else. Figures and the model checkpoint are saved there.
- On Kaggle only, an exit hook copies the executed notebook to `executed_solution.ipynb` in the output folder so it can be downloaded with `kaggle kernels output`.

### 2. Synthetic Cell Segmentation Dataset
- `generate_cell_image(size=64, max_cells=8, min_radius=5, max_radius=12)` draws 3 to 8 circles of random radius, adds 0.3 to 0.5 brightness inside each (overlaps get brighter), adds Gaussian noise (std 0.05) and clips to [0, 1].
- The mask is 1 inside any circle and 0 elsewhere, so overlapping or touching blobs merge into one foreground region. The task is plain foreground/background segmentation; it does not test the paper's separation of touching cells.
- `IMG_SIZE = 64`.

### 3. Data Augmentation
- `CellSegmentationDataset` pre-generates its images. Training items get random horizontal and vertical flips and a random multiple of 90° rotation; validation items are not augmented.
- 100 training images and 25 validation images, batch size 16 (7 training batches per epoch).
- The paper's elastic deformations, shifts and gray-value changes are not implemented.

### 4. U-Net Architecture (from scratch)
- `DoubleConv`: (3×3 conv, padding 1 → BatchNorm → ReLU) × 2. Padded convolutions keep the size, unlike the paper's unpadded ones.
- `DownBlock`: 2×2 max-pool, then `DoubleConv`.
- `UpBlock`: 2×2 transposed convolution that halves the channels, pad to the skip's size if needed, concatenate with the skip, then `DoubleConv`.
- `UNet(in_channels=1, num_classes=2, base_ch=32)`: channels 32 → 64 → 128 → 256 → 512 (half the paper's 64 → 1024), 7,765,442 parameters, and a final 1×1 convolution to 2 classes.

```
Input:            [B, 1, 64, 64]
  inc   (enc 1):  [B, 32, 64, 64]   (skip 1)
  down1 (enc 2):  [B, 64, 32, 32]   (skip 2)
  down2 (enc 3):  [B, 128, 16, 16]  (skip 3)
  down3 (enc 4):  [B, 256, 8, 8]    (skip 4)
  down4 (bottom): [B, 512, 4, 4]
  up1 + skip 4:   [B, 256, 8, 8]
  up2 + skip 3:   [B, 128, 16, 16]
  up3 + skip 2:   [B, 64, 32, 32]
  up4 + skip 1:   [B, 32, 64, 64]
  outc (1×1):     [B, 2, 64, 64]    per-pixel class scores
```

### 5. Loss Functions
- `DiceLoss`: 1 − soft Dice of the foreground probability (smoothing 1.0).
- `CombinedLoss`: cross-entropy with class weights [1.0, 1.5] plus Dice. The paper instead uses cross-entropy weighted by a per-pixel border-weight map.

### 6. Training Loop
- `calculate_iou` gives per-class IoU; the reported score is the foreground IoU averaged over the validation images.
- `train_model` trains for `NUM_EPOCHS = 15` with Adam (lr 1e-3, weight decay 1e-5) and `ReduceLROnPlateau` (factor 0.5, patience 2) on the validation loss, prints the losses, IoU and learning rate each epoch, and saves the checkpoint with the best validation IoU to `OUT_DIR / 'best_unet_model.pth'`.
- **This run:** validation IoU rises from 0.68 after epoch 1 to above 0.99 from epoch 9; the best checkpoint (epoch 13) scores IoU 0.9996, Dice 0.9998 and pixel accuracy 0.9999. The no-skip baseline reaches a best IoU of 0.8517 at epoch 15.

### 7. Results: Training Curves
- Train and validation loss, validation IoU, and the validation − train loss gap per epoch.

### 8. Input / Prediction / Ground Truth Triplets
- Reloads the best checkpoint and shows 6 validation images with their predicted and true masks.

### 9. Intermediate Feature Maps
- For one validation image, shows the first 8 channels and channel 0 of encoder levels 1 to 3, the bottleneck, and decoder levels 1 and 4.

### 10. Quantitative Evaluation
- Reloads the best checkpoint and reports mean foreground IoU, Dice and pixel accuracy on the 25 validation images, with histograms.

### 11. Skip Connections vs No Skip Connections
- `UNetNoSkip` has the same encoder, but each decoder step is a transposed convolution and a `DoubleConv` with no concatenation (6,982,082 parameters).
- It is trained with the same data, loss, optimizer, scheduler and number of epochs (`NOSKIP_EPOCHS = NUM_EPOCHS`), and both validation IoU and loss curves are plotted against the U-Net's.

### 12. Summary
- "What the run above shows" with the run's numbers and caveats, then what was built.

## Deliberate Simplifications vs Full Paper

1. **Data:** synthetic 64×64 blobs instead of the 512×512 EM images and light-microscopy data from the ISBI challenges.
2. **Width:** base width 32 (32 → 512 channels) instead of 64 (64 → 1024).
3. **Padded convolutions:** output size equals input size; the paper uses unpadded convolutions (572×572 input → 388×388 output) and an overlap-tile strategy.
4. **BatchNorm added** after every convolution; the paper uses plain convolution + ReLU.
5. **Loss:** cross-entropy with fixed class weights plus Dice, instead of the paper's border-weight map.
6. **Augmentation:** flips and 90° rotations only; no elastic deformations.
7. **Optimizer:** Adam with batch size 16 instead of SGD with momentum 0.99 and a batch of a single large tile.
8. **Training length:** 15 epochs of 7 batches; the paper trained for about 10 hours on a GPU.
