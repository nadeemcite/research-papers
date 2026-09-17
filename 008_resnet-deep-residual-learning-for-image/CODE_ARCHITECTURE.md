# ResNet — Deep Residual Learning for Image Recognition — Notebook Architecture

## Goal

The notebook implements a residual block and a small ResNet-style network from scratch in PyTorch, trains it on CIFAR-10, and compares it against a plain (non-residual) CNN of the same depth.

## Section-by-section breakdown

### 1. Setup and data loading
- `torch`, `torchvision`, `matplotlib`, `numpy` are installed/verified.
- CIFAR-10 is downloaded and normalized. We use standard train/test splits and a reasonable batch size (128).
- No external datasets beyond torchvision are required.

### 2. Plain building block (baseline)
- A simple stack of two `3×3` convolutional layers with ReLU and batch normalization (kept light; the paper used BN heavily).
- This block is stacked repeatedly to create a plain deep CNN at matched depth.

### 3. Residual block from scratch
- A `ResidualBlock` module with two convolution paths:
  - **Main path:** `Conv → BN → ReLU → Conv → BN`
  - **Shortcut path:** identity when input and output shapes match; otherwise a `1×1` convolution (no bias) with BN to project dimensions.
- The two paths are added element-wise and passed through ReLU: `out = F(x) + shortcut(x)`.
- Stride of 2 in the first conv of a stage halves spatial resolution and doubles channels.

### 4. Plain vs ResNet model builders
- `make_plain_net(block_counts, num_classes)` and `make_resnet(block_counts, num_classes)` construct networks of identical layer counts.
- We choose a small configuration (e.g., `[2,2,2,2]` for an 18-layer-style ResNet) so training completes quickly on a Kaggle GPU while still demonstrating the residual effect.

### 5. Training loop
- Cross-entropy loss, SGD with momentum and weight decay (close to paper settings), step learning-rate decay.
- A reusable `train_epoch`/`evaluate` pair tracks loss and accuracy.
- We intentionally keep the same optimizer and hyperparameters for both networks to isolate the architectural difference.

### 6. Comparison plots
- Train/test accuracy curves for plain net vs. ResNet on the same axes.
- Optional: per-epoch loss curves.
- Expected result: the plain deep net either plateaus earlier or degrades, while the ResNet continues to improve or holds a clear gap.

### 7. What to try next
- Add data augmentation, wider channels, or more stages.
- Replace basic blocks with bottleneck blocks (`1×1 → 3×3 → 1×1`).
- Swap SGD for a cosine scheduler or AdamW.

## Key functions / classes

- `ResidualBlock(in_channels, out_channels, stride=1, projection=None)`
  - Builds main and shortcut paths.
  - `forward(x)` returns `relu(main(x) + shortcut(x))`.
- `make_plain_net(block_counts, num_classes=10)`
  - Sequential plain stack used as a fair baseline.
- `make_resnet(block_counts, num_classes=10)`
  - Same skeleton but replaces plain blocks with `ResidualBlock`.
- `train_epoch(model, loader, criterion, optimizer, device)`
- `evaluate(model, loader, criterion, device)`

## Data flow and shapes

Input image: `(B, 3, 32, 32)`

Initial conv: `(B, 64, 32, 32)`

Stage 1 (stride 1): `(B, 64, 32, 32)` × N blocks

Stage 2 (stride 2): `(B, 128, 16, 16)` × N blocks

Stage 3 (stride 2): `(B, 256, 8, 8)` × N blocks

Stage 4 (stride 2): `(B, 512, 4, 4)` × N blocks

Global average pool + FC: `(B, 512)` → `(B, 10)`

## Deliberate simplifications vs. full paper

- The paper tests on ImageNet with ResNet-34/50/101/152; we use a ResNet-18-style small model on CIFAR-10 so a single GPU run finishes in minutes, not hours.
- We use standard PyTorch BN instead of reimplementing every detail of the paper’s weight initialization.
- We keep a minimal plain-net baseline rather than reproducing VGG exactly.
- The bottleneck block is not included; the basic two-conv block is enough to demonstrate skip connections.
- Full ImageNet-scale training (10-crop testing, scale jitter, 100 epochs, etc.) is omitted for runtime, but the architecture idea is preserved exactly.
