---
title: "pix2pix: Code architecture"
parent: "025 pix2pix"
nav_order: 1
---
# Code Architecture — Pix2Pix (Image-to-Image Translation with Conditional Adversarial Networks)

## Overview

The notebook builds a U-Net generator and a PatchGAN discriminator from scratch in PyTorch and trains them for 4 epochs on 200 edges→shoes pairs at 128×128, on a Kaggle Tesla T4. It is a short demonstration of the mechanics, so the outputs are far from the paper's quality.

## Notebook Structure

### 1. Setup & Imports
- PyTorch, numpy, matplotlib, PIL, plus standard-library helpers.
- Seeds Python, NumPy and PyTorch with 42 and turns on deterministic cuDNN kernels, so a re-run reproduces the summary's numbers.
- Shows figures as JPEG at 80 dpi so the saved notebook stays small.
- Detects Kaggle from its `KAGGLE_KERNEL_RUN_TYPE` environment variable and sets `OUT_DIR`: the current folder on Kaggle (its output folder), `./outputs` anywhere else.
- On Kaggle only, an exit hook copies the executed notebook to `executed_solution.ipynb` in the output folder so it can be downloaded with `kaggle kernels output`.

### 2. Hyperparameters
- `IMG_SIZE = 128`, `BATCH_SIZE = 4`, `LEARNING_RATE = 0.0002`, `BETA1 = 0.5`, `BETA2 = 0.999`, `LAMBDA_L1 = 100`, `EPOCHS = 4`, `NUM_WORKERS = 2`.

### 3. Dataset Preparation
- Downloads `edges2shoes.tar.gz` (about 2 GB) over HTTPS from the Berkeley server with `curl -fsSL`, so a failed download stops the run, and extracts it into the system temp folder.
- `Edge2ShoeDataset` splits each file into left (edges) and right (shoe), resizes both to 143×143 (1.12 × 128), then takes a random 128×128 crop and random mirror for training, or a centre crop for validation, and scales to [-1, 1].
- `MAX_TRAIN = 200`, `MAX_VAL = 30`: only the first 200 of 49,825 training images and 30 of 200 validation images are used. 50 training batches per epoch.

### 4. Generator Architecture (U-Net)
- `DownBlock`: 4×4 conv (stride 2) → optional BatchNorm → LeakyReLU(0.2).
- `UpBlock`: 4×4 transposed conv (stride 2) → BatchNorm → ReLU → optional Dropout, then concatenate the skip.
- `UNet`: 7 down blocks and 6 up blocks plus a final transposed conv with Tanh. `d1` and `d7` have no BatchNorm. Dropout 0.5 on `u1` and `u2`. 41,828,995 parameters.

### 5. Discriminator Architecture (PatchGAN)
- `PatchNet`: concatenates input and target or generated image (6 channels), then conv blocks 64 (no BatchNorm) → 128 → 256 → 512 (stride 1) → a final 1-channel conv. 2,768,641 parameters.
- Outputs raw logits on a 14×14 grid, one per patch with a 70×70 receptive field. There is no Sigmoid: `BCEWithLogitsLoss` applies it.

### 6. Loss Functions & Optimizers
- **Adversarial loss:** `BCEWithLogitsLoss` on the patch grid. **L1 loss:** `L1Loss`, weighted by λ = 100.
- Adam for both networks (lr 0.0002, β1 0.5, β2 0.999).

### 7. Training Loop
- Each batch: one D step on real and fake pairs (`.detach()` on the fake), with the loss halved as in the paper, then one G step on adversarial + λ·L1 loss.
- Prints the epoch averages of D and G and the last batch's adversarial and L1 losses (final epoch: D 0.39, G 32.05, last-batch L1 0.23). After each epoch, `save_sample` shows one fixed validation example.

### 8. Loss Curves
- G and D average losses per epoch (epochs 1 to 4).

### 9. Final Results
- `generate_test_grid` shows 8 random validation examples: input edges, generated output, real target.

### 10. Summary
- "What the run above shows" with the run's numbers, then a table of what was built.

## Key Classes and Functions

| Component | Description |
|-----------|-------------|
| `Edge2ShoeDataset` | Loads paired images, splits, augments |
| `DownBlock`, `UpBlock` | Encoder and decoder building blocks |
| `UNet` | Generator: 7 down + 7 up steps with skips, dropout on `u1` and `u2` |
| `PatchNet` | Discriminator: 70×70 PatchGAN returning logits |
| `save_sample()` | One input / output / target figure per epoch |
| `generate_test_grid()` | Grid of 8 validation examples |

## Data Flow / Shapes

```
Input: edges (B, 3, 128, 128)
  → UNet encoder: 128→64→32→16→8→4→2→1 (spatial), channels 64→128→256→512→512→512→512
  → UNet decoder: 1→2→4→8→16→32→64→128 (spatial) with skip concatenation
  → Output: (B, 3, 128, 128) in [-1, 1]

PatchNet input: cat(input, target_or_fake) → (B, 6, 128, 128)
  → 4 conv blocks + output conv → (B, 1, 14, 14) patch logits
```

## Deliberate Simplifications vs Full Paper

1. **Image size:** 128×128 instead of the paper's 256×256, so the generator has 7 down/up steps instead of 8.
2. **Dataset size:** 200 pairs instead of the full edges2shoes training set (49,825 images here).
3. **Training length:** 4 epochs at batch size 4 (200 steps). For edges→shoes the paper also uses batch size 4, but trains 15 epochs on about 50,000 images (Appendix 6.2).
4. **Evaluation mode:** `save_sample` and `generate_test_grid` call `netG.eval()`, which turns dropout off and uses BatchNorm's running statistics. The paper keeps dropout on and uses batch statistics at test time.
5. **No FCN-score evaluation or AMT study:** the notebook only shows samples.
6. **Single task:** edges→shoes only, whereas the paper tests many tasks.
