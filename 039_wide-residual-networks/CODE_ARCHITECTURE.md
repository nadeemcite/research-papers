---
title: "Wide ResNet: Code architecture"
parent: "039 Wide ResNet"
nav_order: 1
---
# Wide Residual Networks — Notebook Architecture

## Goal

The notebook implements the paper's WRN-n-k architecture from scratch in PyTorch and trains seven models on **real CIFAR-10** on a single GPU (a Kaggle T4) with one identical recipe. It compares networks that grow *wider* (WRN-16-1/2/4/8) with networks that grow *deeper* (WRN-16/28/40/52-1), then runs a lesion study (switch off one residual block at a time) and shows the best model's predictions. Dropout is implemented in the block but not used in the main comparison.

## Section-by-section breakdown

### 1. Setup
- Imports `torch`, `torchvision`, `numpy`, `matplotlib`; picks the GPU if present and prints its name; sets `cudnn.benchmark`.
- One shared recipe: `EPOCHS = 20`, `BATCH_SIZE = 256`, `PEAK_LR = 0.2`, weight decay `5e-4`, Nesterov momentum `0.9`, label smoothing `0.1`, mixed precision on CUDA.
- `WIDE_CONFIGS = [(16,1), (16,2), (16,4), (16,8)]` and `DEEP_CONFIGS = [(16,1), (28,1), (40,1), (52,1)]`.

### 2. CIFAR-10 on the GPU
- `load_cifar(train)` reads CIFAR-10 with `torchvision.datasets.CIFAR10`. On Kaggle, if the dataset `pankrzysiu/cifar10-python` is attached, it is read from `/kaggle/input/cifar10-python` (about 3 seconds); otherwise torchvision downloads it into `./data`, which can take many minutes from a slow server.
- The 50,000 + 10,000 images are stored as `uint8` tensors on the GPU. `normalize` applies the per-channel mean and standard deviation from the training set.
- `augment(x)` does the paper's augmentation in a batched way on the GPU: 4-pixel reflect padding, random 32×32 crop, random horizontal flip. A figure shows training images next to their augmented versions.

### 3. The WRN-n-k architecture
- `WideBasicBlock(c_in, c_out, stride, dropout)` is the pre-activation block: BN → ReLU → conv3×3 → (dropout) → BN → ReLU → conv3×3, with a 1×1 projection shortcut when the shape changes. An `enabled` flag lets the lesion study keep only the shortcut path.
- `WideResNet(depth, k, num_classes, dropout)` builds an initial 3×3 conv with 16 channels, then three groups of `N = (depth − 4) / 6` blocks with `16k`, `32k`, `64k` channels (strides 1, 2, 2), then BN-ReLU, global average pooling and a linear layer. Convolutions use Kaiming initialisation.
- `count_params(model)` and a table print the parameters and block count of all seven models (0.18M to 10.96M).

### 4. Training recipe
- `evaluate(model, bs)` returns the test accuracy and the logits.
- `train(depth, k, epochs, dropout, seed)` trains with SGD (Nesterov), a `OneCycleLR` schedule, label smoothing and mixed precision. It records test accuracy and **wall-clock training time** (evaluation excluded) after every epoch, and prints progress every 5 epochs.

### 5. The race: wider vs. deeper
- A loop trains every distinct configuration once and stores the results and models.
- A three-panel figure plots accuracy against parameters, accuracy against total training time, and the learning curves against wall-clock time.
- A head-to-head table compares WRN-16-2 with WRN-40-1 and WRN-52-1 (similar parameter counts): accuracy difference and speed-up.

### 6. Lesion study
- `lesion(model)` switches off each residual block in turn (keeping its shortcut) and measures the accuracy drop on the test set.
- Bar charts for WRN-52-1 (24 blocks) and WRN-16-8 (6 blocks), with dotted lines at the resolution changes, and a printed count of blocks whose removal costs under 1 point. This follows the spirit of Veit et al. (2016) and is a way to look at the paper's diminishing feature reuse idea.

### 7. What the best network sees
- A grid of 24 random test predictions, the 16 most confident mistakes, and a normalised confusion matrix. The printout gives per-class accuracy and the most common confusion.

### 8. Summary
- Text with the simplifications and the results of the saved run, and a printed summary table of accuracy and training time for all models.

## Key functions / classes

- `WideBasicBlock(c_in, c_out, stride, dropout)`: the paper's `B(3,3)` block, with an `enabled` switch.
- `WideResNet(depth, k, num_classes, dropout)`: the full WRN builder.
- `count_params(model)`: trainable parameter count.
- `load_cifar(train)`, `normalize(x_u8)`, `augment(x)`: GPU data pipeline.
- `evaluate(model, bs)`, `train(depth, k, epochs, dropout, seed)`: evaluation and the training recipe.
- `lesion(model)`: per-block removal study.

## Data flow and shapes

For WRN-16-2 (`k = 2`, `N = 2` blocks per group):

- Input: `(B, 3, 32, 32)`
- Initial conv: `(B, 16, 32, 32)`
- Group 1 (stride 1): `(B, 32, 32, 32)`, 2 blocks
- Group 2 (stride 2): `(B, 64, 16, 16)`, 2 blocks
- Group 3 (stride 2): `(B, 128, 8, 8)`, 2 blocks
- BN-ReLU, global average pool: `(B, 128)`
- Linear layer: `(B, 128)` → `(B, 10)`

Depth follows the paper's convention `n = 6N + 4`, so WRN-16 has `N = 2` blocks per group (6 blocks in total) and WRN-52 has `N = 8` (24 blocks).

## Deliberate simplifications vs. the paper

- **20 epochs, not 200.** The paper uses step decay (×0.2 at epochs 60, 120, 160), batch size 128 and learning rate 0.1. The notebook uses a one-cycle schedule with peak 0.2, batch 256 and label smoothing so that seven models fit in about 25 minutes. The accuracies are lower than the paper's.
- **Mixed precision and GPU-side augmentation** for speed.
- **Largest model WRN-16-8.** The paper's best CIFAR-10 model is WRN-28-10 (3.89% error with dropout).
- **No dropout in the main comparison,** and no SVHN, COCO or ImageNet experiments.
- **One seed per model.** The paper reports medians over 5 runs; differences of a few tenths of a point here are noise.
- **No ResNet-1001 baseline,** so the paper's comparison with very deep thin networks is not reproduced. The notebook tests the width-vs-depth idea at small scale.
- **No convolution-type ablations** (B(3,1,3), B(3,3,3), and so on).
