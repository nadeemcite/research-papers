---
title: "DDIM: Code architecture"
parent: "085 DDIM"
nav_order: 1
---
# Code Architecture — DDIM

## Overview

The notebook trains a small noise-prediction U-Net with the DDPM objective on FashionMNIST (T = 1000) and then samples it two ways: DDIM, a deterministic process on 50 of the 1000 timesteps, and DDPM, the stochastic process on all 1000. It compares their speed and samples, and tests DDIM's properties from the paper: consistency across step counts, interpolation in x_T, the effect of η, and encoding real images to x_T and decoding them back.

## Section-by-Section Breakdown

### 1. Setup and Imports
- PyTorch, torchvision, NumPy, matplotlib; no tqdm
- Seeds 42; deterministic cuDNN/cuBLAS settings; runs on a Kaggle GPU
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Hyperparameters
- `T = 1000`, `BETA_START = 1e-4`, `BETA_END = 0.02`, `IMG_SIZE = 32`, `CHANNELS = 1`
- `BATCH_SIZE = 128`, `EPOCHS = 20`, `LR = 2e-4`, `BASE_CH = 32`, `EMA_DECAY = 0.999`
- `DDIM_STEPS = 50`

### 3. Forward Diffusion Process
- Linear β schedule and the precomputed α_t, ᾱ_t, √ᾱ_t, √(1 − ᾱ_t), 1/√α_t and posterior variance
- `q_sample(x_0, t, noise)`: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε; prints ᾱ at the first and last step

### 4. Dataset and DataLoader
- All 60,000 FashionMNIST training images, downloaded quietly, resized to 32×32 and scaled to [−1, 1], transformed once into one tensor; a `DataLoader` with `num_workers=0` and a seeded shuffle (468 batches per epoch)

### 5. Visualize Forward Diffusion
- One image noised at t = 0, 50, 100, 200, 400, 600, 800 and 999

### 6. U-Net Denoiser
- `SinusoidalPositionEmbeddings(128)` → MLP (128 → 256 → 128): the timestep embedding
- `ResBlock`: GroupNorm → SiLU → 3×3 conv, plus the projected time embedding, then GroupNorm → SiLU → 3×3 conv, with a skip connection
- `SelfAttention`: spatial multi-head attention (4 heads), `einsum('bhdn,bhem->bhnm')` for the scores, softmax over the attended positions, `einsum('bhnm,bhdm->bhdn')` for the output, 1×1 projection and residual
- `DownBlock` (two ResBlocks, optional attention, stride-2 conv) and `UpBlock` (transposed conv, concat skip, two ResBlocks, optional attention)
- `UNet`: 32×32 (32 ch) → 16×16 (64 ch) → 8×8 (128 ch) bottleneck → back up; attention at 16×16 on the way down and up; 2,038,243 parameters
- `ema_model`: a frozen copy updated during training

### 7. Training Loop
- Random t in [0, 1000), noise ε, x_t = `q_sample`, loss = MSE(ε_θ(x_t, t), ε); Adam; EMA update after every step; prints the average loss and elapsed time per epoch

### 8. Loss Curve
- Average loss per epoch

### 9. DDIM Sampling
- `ddim_sample(model, x_T, ddim_steps, eta)`: timesteps 0, T/S, 2T/S, ... (for S = 50: 0, 20, ..., 980), run in reverse; each step predicts x̂_0 = (x_t − √(1 − ᾱ_t) ε_θ)/√ᾱ_t and sets x_prev = √ᾱ_prev x̂_0 + √(1 − ᾱ_prev − σ²) ε_θ + σ z, with σ from Eq. (16) (σ = 0 for η = 0)
- `ddim_invert(model, x_0, ddim_steps)`: the same deterministic update run forward on the same timesteps, x_0 → x_T (the encoding of Section 5.4)

### 10. DDPM Sampling
- `p_sample` / `p_sample_loop`: stochastic ancestral sampling through all 1000 steps with the posterior variance

### 11. Speed Comparison
- 64 samples from the same x_T with DDIM (50 steps) and DDPM (1000 steps), using the EMA model, timed with CUDA synchronization; prints both times, the speed-up and simple pixel statistics against 64 real images

### 12. Visualize DDIM vs DDPM Samples
- 8 samples from each method

### 13. DDIM Consistency
- 4 values of x_T decoded with 10, 20, 50, 100 and 1000 steps; prints the MSE (pixels in [0, 1]) of each to the 1000-step result, and between 1000-step samples from different x_T for scale

### 14. Latent Space Interpolation
- `slerp` between two x_T, decoded with 50 DDIM steps at 8 points

### 15. Effect of η
- The same x_T decoded with η = 0, 0.3, 0.7 and 1.0

### 16. Reconstruction Test
- 8 real images encoded with `ddim_invert` and decoded with `ddim_sample`, with 50 and with 200 steps; prints the reconstruction MSE (pixels in [0, 1], as in the paper's Table 2) and shows originals and reconstructions

### 17. Summary
- A printed summary of the settings, times and speed-up, then `### What the run above shows` with the saved run's real numbers

## Key Functions

| Function | Purpose |
|---|---|
| `q_sample(x_0, t, noise)` | Forward noising in one step |
| `UNet`, `SelfAttention`, `ResBlock` | Noise predictor ε_θ(x_t, t) |
| `ddim_sample(model, x_T, steps, eta)` | DDIM sampling on a timestep sub-sequence |
| `ddim_invert(model, x_0, steps)` | Deterministic DDIM encoding x_0 → x_T |
| `p_sample_loop(model, x_T)` | DDPM sampling through all T steps |
| `slerp(t, v0, v1)` | Spherical interpolation between two x_T |

## Data Flow / Shapes

```
training:  x_0 (128, 1, 32, 32), t (128,) → q_sample → x_t → U-Net → ε_θ (128, 1, 32, 32) → MSE with ε
DDIM:      x_T (64, 1, 32, 32) → 50 deterministic steps (t = 980, 960, ..., 0) → x_0
DDPM:      x_T (64, 1, 32, 32) → 1000 stochastic steps → x_0
encoding:  x_0 → ddim_invert (t = 0 → 980) → x_T → ddim_sample → reconstructed x_0
```

## Deliberate Simplifications vs. the Paper

1. **Data and model:** 32×32 grayscale FashionMNIST and a 2.0M-parameter U-Net, instead of CIFAR10, CelebA and LSUN with the DDPM authors' larger models.
2. **Training:** 20 epochs (9,360 steps) with EMA decay 0.999, instead of the pretrained DDPM checkpoints the paper samples from.
3. **Timesteps:** an evenly spaced sub-sequence (0, 20, ..., 980), one of the options in the paper (it also uses a quadratic spacing for some datasets).
4. **Evaluation:** sample grids, timing, pixel statistics and MSE instead of FID.
