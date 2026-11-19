---
title: "DDPM: Code architecture"
parent: "084 DDPM"
nav_order: 1
---
# Code Architecture — DDPM

## Overview

The notebook implements DDPM from scratch in PyTorch: the forward noising process, a compact U-Net that predicts the noise, training with the paper's simple loss plus an EMA copy of the weights, and 1000-step ancestral sampling. It trains on all 60,000 FashionMNIST training images resized to 32×32 on a Kaggle GPU, then shows 64 samples and compares simple pixel statistics with real images.

## Section-by-Section Breakdown

### 1. Setup and Imports
- PyTorch, torchvision, NumPy, matplotlib; no tqdm
- Seeds PyTorch and NumPy with `SEED = 42`; deterministic cuDNN and cuBLAS settings (`CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True, warn_only=True)`)
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Hyperparameters
- `T = 1000`, `BETA_START = 1e-4`, `BETA_END = 0.02` (as in the paper)
- `IMG_SIZE = 32`, `CHANNELS = 1`, `BATCH_SIZE = 128`, `LR = 2e-4`, `EPOCHS = 25`
- `BASE_CH = 32`, `GROUPS = 8` (GroupNorm groups), `EMA_DECAY = 0.999`

### 3. Forward Diffusion Process
- `linear_beta_schedule` and the precomputed terms: α_t = 1 − β_t, ᾱ_t (cumulative product), √ᾱ_t, √(1 − ᾱ_t), 1/√α_t, and the posterior variance β̃_t = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t)
- `q_sample(x_0, t, noise)`: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε in one step
- Prints ᾱ_T = 4.04e-05, so x_T is almost pure noise

### 4. Dataset and DataLoader
- FashionMNIST train split, downloaded quietly to a temp folder; `Resize(32)`, `ToTensor`, `Normalize(0.5, 0.5)` to [−1, 1]
- All 60,000 images are transformed once into one tensor (60000, 1, 32, 32); a `DataLoader` over a `TensorDataset` with `num_workers=0`, a seeded shuffle and `drop_last=True` gives 468 batches per epoch

### 5. Visualize Forward Diffusion
- One image noised at t = 0, 50, 100, 200, 400, 600, 800 and 999

### 6. U-Net Denoiser Architecture
- `SinusoidalPositionEmbeddings(32)` → `Linear(32, 128)` → SiLU → `Linear(128, 128)`: the timestep embedding
- `ResBlock`: GroupNorm → SiLU → 3×3 conv, then add the projected time embedding, then GroupNorm → SiLU → 3×3 conv, plus a skip (1×1 conv when channels change)
- `AttentionBlock`: spatial multi-head self-attention (4 heads); `attn = einsum('bhdn,bhdm->bhnm', q, k) · scale`, softmax over the attended positions m, then `einsum('bhnm,bhdm->bhdn', attn, v)`, a 1×1 projection and a residual connection
- `Downsample`: 3×3 conv with stride 2; `Upsample`: nearest-neighbour ×2, then 3×3 conv
- `UNet`, channels 32 → 64 → 128:
  - Encoder: ResBlock(32) at 32×32 → down → ResBlock(32→64) + attention at 16×16 → down → ResBlock(64→128) at 8×8
  - Bottleneck: two ResBlocks at 8×8 (128 channels)
  - Decoder: concat skip, ResBlock(256→128) at 8×8 → up → concat, ResBlock(192→64) + attention at 16×16 → up → concat, ResBlock(96→32) at 32×32
  - Output: GroupNorm → SiLU → 3×3 conv to 1 channel (the predicted noise)
- 1,951,329 parameters; `ema_model` is a frozen copy updated during training

### 7. Training Loop
- Per batch: random t in [0, T), noise ε, x_t = `q_sample`, loss = `F.mse_loss(model(x_t, t), ε)` (the paper's unweighted L_simple), Adam step, then EMA update `p_ema = 0.999 · p_ema + 0.001 · p`
- Prints only the average loss and elapsed time per epoch

### 8. Loss Curve
- Average loss per epoch

### 9. Sampling (Reverse Diffusion Process)
- `p_sample`: μ = (1/√α_t) · (x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t)); for t > 0 adds √β̃_t · z with z ~ N(0, I), at t = 0 returns μ
- `sample`: starts from x_T ~ N(0, I) and runs all 1000 steps with the EMA model, for 64 images
- Prints the pixel mean, standard deviation and share of near-black pixels for the samples and for 64 real images

### 10. Visualize Generated Samples
- 8×8 grid of the 64 samples

### 11. Summary
- `### What the run above shows`: the saved run's real losses, an honest description of the samples, and the statistics

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `linear_beta_schedule` | β_t from 1e-4 to 0.02 over T steps |
| `q_sample` | Closed-form noising x_0 → x_t |
| `SinusoidalPositionEmbeddings` | Timestep embedding |
| `ResBlock` | Residual conv block with the time embedding added |
| `AttentionBlock` | Spatial self-attention (used at 16×16) |
| `UNet` | Noise predictor ε_θ(x_t, t) |
| `p_sample`, `sample` | One reverse step, and the full 1000-step sampler |

## Data Flow / Shapes

```
x_0 (128, 1, 32, 32) in [−1, 1], t (128,), ε (128, 1, 32, 32)
x_t = q_sample(x_0, t, ε)
UNet: 32×32×32 → 16×16×64 (+attention) → 8×8×128 → bottleneck 8×8×128
      → 8×8×128 → 16×16×64 (+attention) → 32×32×32 → ε_θ (128, 1, 32, 32)
loss = MSE(ε_θ, ε)
sampling: x_999 ~ N(0, I) → p_sample × 1000 → x_0 (64, 1, 32, 32)
```

## Deliberate Simplifications vs. the Paper

1. **Data:** 32×32 grayscale FashionMNIST instead of CIFAR10, CelebA-HQ and LSUN.
2. **Model size:** 1.95M parameters with one ResBlock per level and channels 32-64-128; the paper's CIFAR10 U-Net has 35.7M parameters.
3. **Training length:** 25 epochs (11,700 steps) instead of 800k steps, and no dropout (the paper used 0.1 on CIFAR10).
4. **EMA decay:** 0.999 instead of 0.9999, because the run is much shorter.
5. **Sampling variance:** the posterior variance β̃_t; the paper found β_t and β̃_t gave similar results.
6. **Evaluation:** a sample grid and simple pixel statistics instead of FID and Inception score.
