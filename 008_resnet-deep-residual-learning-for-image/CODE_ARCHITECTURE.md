---
title: "ResNet: Code architecture"
parent: "008 ResNet"
nav_order: 1
---
# ResNet — Deep Residual Learning for Image Recognition — Notebook Architecture

## Goal

The notebook implements a residual block and a small ResNet-style network from scratch in PyTorch, trains it on CIFAR-10, and compares it against a plain (non-residual) CNN of the same depth.

## Section-by-section breakdown

### 1. Setup and data loading
- `torch`, `torchvision`, `matplotlib` and `numpy` are imported.
- CIFAR-10 is downloaded through torchvision (progress bars silenced) and normalized, with random-crop and horizontal-flip augmentation on the training set. Standard train/test splits, batch size 128.
- The download is the slow part of a fresh run; the training itself took about 17 to 18 minutes per model on a Kaggle GPU.

### 2. Plain building block (baseline)
- `ConvBlock`: two `3×3` convolutions, each followed by batch normalization and ReLU.
- `PlainNet` stacks four stages of `ConvBlock`s with no shortcuts.

### 3. Residual block from scratch
- A `ResidualBlock` module with two paths:
  - **Main path:** `Conv → BN → ReLU → Conv → BN`
  - **Shortcut path:** identity when input and output shapes match; otherwise a `1×1` convolution (no bias, no batch normalization) to project dimensions. The shortcut is chosen automatically from the stride and channel counts.
- The two paths are added element-wise and passed through ReLU: `out = F(x) + shortcut(x)`.
- Stride of 2 in the first conv of a stage halves spatial resolution and doubles channels.

### 4. Plain vs ResNet model builders
- `PlainNet(block_counts, num_classes)` and `ResNet(block_counts, num_classes)` are two classes with the same skeleton and the same layer counts.
- Both use `block_counts=(2,2,2,2)`, an 18-layer-style configuration of about 11M parameters each.

### 5. Training loop
- 25 epochs of cross-entropy loss with SGD (learning rate 0.1, momentum 0.9, weight decay 5e-4) and a step learning-rate decay (×0.2 every 10 epochs).
- A reusable `train_epoch`/`evaluate` pair tracks loss and accuracy.
- We intentionally keep the same optimizer and hyperparameters for both networks to isolate the architectural difference.

### 6. Comparison plots
- Train/test accuracy curves for plain net vs. ResNet on the same axes.
- Per-epoch loss curves in a second panel.
- Result: the two networks end tied (PlainNet 91.18%, ResNet 91.08% test accuracy). At 18 layers with batch normalization there is no degradation gap; PlainNet starts faster in the first epoch and ResNet catches up by about epoch 15.

### 7. What to try next
- Try CIFAR-style 20- and 56-layer networks to look for the degradation gap the paper reports.
- Replace basic blocks with bottleneck blocks (`1×1 → 3×3 → 1×1`).
- Swap SGD for a cosine scheduler or AdamW.

## Key functions / classes

- `ResidualBlock(in_ch, out_ch, stride=1)`
  - Builds the main path, plus a `1×1` projection shortcut when the stride or channel count changes.
  - `forward(x)` returns `relu(main(x) + shortcut(x))`.
- `ConvBlock(in_ch, out_ch, stride=1)`
  - The plain two-convolution block used by `PlainNet`.
- `PlainNet(block_counts, num_classes=10)`
  - Stack of `ConvBlock`s, used as the baseline.
- `ResNet(block_counts, num_classes=10)`
  - Same skeleton but with `ResidualBlock`s.
- `train_epoch(model, loader, criterion, optimizer, device)`
- `evaluate(model, loader, criterion, device)`
- `count_parameters(model)`

## Data flow and shapes

Input image: `(B, 3, 32, 32)`

Initial conv: `(B, 64, 32, 32)`

Stage 1 (stride 1): `(B, 64, 32, 32)` × N blocks

Stage 2 (stride 2): `(B, 128, 16, 16)` × N blocks

Stage 3 (stride 2): `(B, 256, 8, 8)` × N blocks

Stage 4 (stride 2): `(B, 512, 4, 4)` × N blocks

Global average pool + FC: `(B, 512)` → `(B, 10)`

## Deliberate simplifications vs. full paper

- The paper tests on ImageNet with ResNet-34/50/101/152; we use a ResNet-18-style small model on CIFAR-10 so a single GPU run finishes in under an hour (about 17 to 18 minutes per model on Kaggle).
- We use standard PyTorch BN instead of reimplementing every detail of the paper’s weight initialization.
- We keep a minimal plain-net baseline rather than reproducing VGG exactly.
- The bottleneck block is not included; the basic two-conv block is enough to demonstrate skip connections.
- Full ImageNet-scale training (10-crop testing, scale jitter, 100 epochs, etc.) is omitted for runtime, but the architecture idea is preserved exactly.
