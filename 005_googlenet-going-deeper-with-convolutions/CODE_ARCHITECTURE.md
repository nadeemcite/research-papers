# CODE_ARCHITECTURE.md — GoogLeNet / Inception Module

This notebook builds a **minimal but faithful Inception-style network** from scratch in PyTorch, then compares its parameter count against a plain CNN of similar depth.

## Notebook sections

### 1. Setup and dataset
- Install standard packages (`torch`, `torchvision`, `matplotlib`).
- Load CIFAR-10 (32×32 colour images, 10 classes) and normalize to `[-1, 1]`.
- Define small training/validation dataloaders.

### 2. Inception building block
- `ConvBNReLU(in_ch, out_ch, kernel, stride, padding)` — a tiny wrapper for conv + batch norm + ReLU.
- `InceptionModule(in_ch, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj)`:
  - **1×1 branch:** `ConvBNReLU(in_ch → ch1x1, k=1)`.
  - **3×3 branch:** `ConvBNReLU(in_ch → ch3x3_reduce, k=1)` → `ConvBNReLU(ch3x3_reduce → ch3x3, k=3, pad=1)`.
  - **5×5 branch:** `ConvBNReLU(in_ch → ch5x5_reduce, k=1)` → `ConvBNReLU(ch5x5_reduce → ch5x5, k=5, pad=2)`.
  - **Pool branch:** `MaxPool2d(k=3, stride=1, pad=1)` → `ConvBNReLU(in_ch → pool_proj, k=1)`.
  - Concatenate the four outputs along `dim=1`. All branches preserve spatial size.

### 3. Mini-GoogLeNet
Because the full 22-layer GoogLeNet is large and needs 224×224 inputs, this notebook builds a scaled-down version for CIFAR-10:
- Stem: `ConvBNReLU(3 → 64, k=3, pad=1)` → maxpool 2×2/2.
- Stage 1: two Inception modules with increasing 3×3/5×5 capacity.
- Downsample: maxpool 2×2/2.
- Stage 2: two more Inception modules.
- Global average pooling → dropout → linear → softmax.
No auxiliary classifiers are included — they are omitted as a deliberate simplification because CIFAR-10 is small and the network is shallow.

### 4. Plain baseline CNN
- Same number of stages and target output channels, but no parallel branches or 1×1 bottlenecks. Built from stacked 3×3 conv + BN + ReLU blocks.

### 5. Training loop
- Cross-entropy loss, Adam optimizer, small learning rate.
- Trains for a few epochs on CPU/GPU with progress prints.
- Tracks train/validation loss and accuracy.

### 6. Analysis cell
- Print total trainable parameters of Inception model and plain CNN.
- Plot per-epoch loss/accuracy curves.
- Print a sample of predictions.

## Data flow / shapes

```
Input image:  [B, 3, 32, 32]
Stem:         [B, 64, 16, 16]  (conv + maxpool)
Stage 1 A:    [B, 256, 16, 16]
Stage 1 B:    [B, 480, 16, 16]
Pool:         [B, 480, 8, 8]
Stage 2 A:    [B, 512, 8, 8]
Stage 2 B:    [B, 512, 8, 8]
GlobalAvgPool:[B, 512, 1, 1]
Classifier:   [B, 10]
```

## Deliberate simplifications vs. the full paper

| Paper detail | Simplification in notebook | Why |
|---|---|---|
| 224×224 ImageNet input | 32×32 CIFAR-10 input | Faster Kaggle validation; still exercises all architectural ideas. |
| Full 22-layer GoogLeNet (9 Inception modules, two classifiers) | 4 Inception modules, one classifier | Fits CIFAR-10 scale and runtime limits; module math is identical. |
| Auxiliary classifiers | Omitted | Shallow mini-network and small dataset make vanishing gradients unlikely. |
| 7×7/2 initial conv and two early 1×1/3×3 stem convolutions | Single 3×3 stem + one maxpool | CIFAR-10 resolution does not need the original large-field stem. |
| Exact GoogLeNet channel counts | Scaled down but same branching ratios | Keeps the design pattern while matching memory constraints. |
| Local Response Normalization | Replaced by BatchNorm | BatchNorm is the modern equivalent and more stable to train from scratch. |
