---
title: "DCGAN: Code architecture"
parent: "023 DCGAN"
nav_order: 1
---
# Code Architecture — DCGAN Notebook

## Overview

This notebook implements the DCGAN architecture from Radford et al. (2015) following the paper's architectural guidelines. A convolutional generator and discriminator are trained adversarially on CIFAR-10 (RGB, resized to 64×64) for 15 epochs on a Kaggle Tesla T4, then the generator is probed with sample grids, a latent-space interpolation walk and a noise-averaging demo.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision, matplotlib, numpy, plus `json`, `time` and a few standard-library helpers.
- Seeds PyTorch and NumPy with 42, turns on deterministic cuDNN kernels so a re-run reproduces the quoted numbers, and picks CUDA when available (the Kaggle run used a Tesla T4).
- Shows figures as JPEG at 80 dpi so the saved notebook stays small.
- On Kaggle only, registers an exit hook that copies the executed notebook (with outputs) to `/kaggle/working/executed_solution.ipynb`, so it can be downloaded with `kaggle kernels output`.

### 2. Hyperparameters
- `latent_dim = 100`: noise vector size (as in the paper).
- `img_channels = 3`, `img_size = 64`: 64×64 RGB output, the paper's resolution.
- `feature_maps_g = 128`: generator base width, so its first layer has 128×8 = 1024 channels at 4×4, as in the paper.
- `feature_maps_d = 64`: discriminator base width (64 → 512 channels).
- `lr = 0.0002`, `beta1 = 0.5`: the paper's Adam settings.
- `batch_size = 128`: as in the paper.
- `num_epochs = 15`: 390 batches per epoch, 5,850 iterations in total, far fewer than the paper's ~117K (5 epochs on 3M LSUN images), to fit Kaggle's time limit.
- `leak_slope = 0.2`: LeakyReLU slope in the discriminator.

### 3. Data Loading
- Loads the CIFAR-10 training set (50,000 images) from the attached Kaggle dataset `pankrzysiu/cifar10-python` at `/kaggle/input/cifar10-python` when present; elsewhere torchvision downloads it to `./data` with the progress bar silenced.
- `Resize(64)` → `ToTensor()` → `Normalize(0.5, 0.5)` per channel, scaling pixels to [-1, 1] to match the generator's Tanh.
- `DataLoader` with `shuffle=True`, `drop_last=True`, `num_workers=2`: 390 batches per epoch.

**Data flow:** CIFAR-10 32×32 RGB → Resize(64×64) → Normalize([-1, 1]) → tensor `[128, 3, 64, 64]`

### 4. Generator Network (G)

`z` is shaped `[B, 100, 1, 1]`, so the paper's "project and reshape" first layer is a `ConvTranspose2d` with kernel 4, stride 1, pad 0 (equivalent to a fully connected layer reshaped to 4×4).

| Layer | Op | Output shape |
|---|---|---|
| 1 | ConvTranspose2d(100 → 1024, k4, s1, p0) + BatchNorm + ReLU | `[B, 1024, 4, 4]` |
| 2 | ConvTranspose2d(1024 → 512, k4, s2, p1) + BatchNorm + ReLU | `[B, 512, 8, 8]` |
| 3 | ConvTranspose2d(512 → 256, k4, s2, p1) + BatchNorm + ReLU | `[B, 256, 16, 16]` |
| 4 | ConvTranspose2d(256 → 128, k4, s2, p1) + BatchNorm + ReLU | `[B, 128, 32, 32]` |
| 5 | ConvTranspose2d(128 → 3, k4, s2, p1) + Tanh | `[B, 3, 64, 64]` |

- 12,658,432 parameters.
- No BatchNorm on the output layer, ReLU everywhere except the Tanh output, no pooling, no fully connected hidden layers.

### 5. Discriminator Network (D)

| Layer | Op | Output shape |
|---|---|---|
| 1 | Conv2d(3 → 64, k4, s2, p1) + LeakyReLU(0.2), no BatchNorm | `[B, 64, 32, 32]` |
| 2 | Conv2d(64 → 128, k4, s2, p1) + BatchNorm + LeakyReLU(0.2) | `[B, 128, 16, 16]` |
| 3 | Conv2d(128 → 256, k4, s2, p1) + BatchNorm + LeakyReLU(0.2) | `[B, 256, 8, 8]` |
| 4 | Conv2d(256 → 512, k4, s2, p1) + BatchNorm + LeakyReLU(0.2) | `[B, 512, 4, 4]` |
| 5 | Conv2d(512 → 1, k4, s1, p0) + Sigmoid | `[B, 1, 1, 1]` |

- 2,765,568 parameters.
- No BatchNorm on the input layer, strided convolutions instead of pooling, a single sigmoid output.

### 6. Weight Initialization
- `weights_init` draws every Conv/ConvTranspose weight from N(0, 0.02) and BatchNorm weights from N(1, 0.02) with zero bias. The notebook prints the first generator layer's weight std (0.0200) as a check.

### 7. Loss Function & Optimizers
- `nn.BCELoss` with labels real = 1, fake = 0.
- D loss = BCE(D(real), 1) + BCE(D(G(z)), 0); G loss = BCE(D(G(z)), 1), the non-saturating form from the original GAN paper.
- Two Adam optimizers, both `lr=0.0002`, `betas=(0.5, 0.999)`.

### 8. Training Loop
For each batch:
1. **Discriminator step:** real batch with label 1, a fresh fake batch (`G(z).detach()`) with label 0, backprop both, step D.
2. **Generator step:** the same fake batch through the updated D with label 1, backprop, step G.
3. Record both losses (5,850 values each).

After epochs 1, 3, 6, 9, 12 and 15 it prints Loss_D, Loss_G, D(x) and D(G(z)) for the last batch and saves samples from a fixed batch of 64 noise vectors. At the end it prints the training time and writes the run's key numbers to `dcgan_results.json`.

**This run:** 30 to 38 minutes of training across runs (the losses are identical from run to run). Last-epoch means: Loss_D 0.57, Loss_G 3.70 (last batch: 0.06 and 3.84, D(x) = 0.98, D(G(z)) = 0.03).

### 9. Loss Curves
- Both losses per iteration, with a dashed line at log(2) ≈ 0.69 for reference (G's loss at the theoretical equilibrium; D's would be 2·log(2) ≈ 1.39).
- In this run the losses oscillate and do not converge: Loss_G spikes to about 24 early, then swings mostly between 1 and 7, and from about iteration 3,900 the spikes grow while Loss_D repeatedly drops near 0, so the discriminator ends ahead.

### 10. Generated Samples Over Training
- One 8×8 grid per saved checkpoint (epochs 1, 3, 6, 9, 12, 15), all from the same fixed noise.

### 11. Real vs Fake Comparison
- 32 real CIFAR-10 images next to 32 fresh samples. The samples are blurry, with CIFAR-like colours and layouts but no clearly recognisable objects.

### 12. Latent Space Walk (Interpolation)
- Linearly interpolates between two random `z` vectors in 10 steps (the paper's Section 6.1 experiment). In this run the transition is gradual, with no sudden jumps.

### 13. Latent Space Vector Arithmetic (averaging demo)
- Compares 8 images from single noise vectors with 8 images from noise vectors averaged over 3 samples each. This only illustrates the averaging step behind the paper's face arithmetic (Section 6.3.2); the notebook does no concept arithmetic itself.

### 14. Summary
- Recap, a "What the run above shows" note with the numbers above, and a paper-vs-notebook table.

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Dataset | LSUN bedrooms (3M), ImageNet-1k, faces (350K) | CIFAR-10 train (50K RGB), resized from 32×32 to 64×64 |
| Training length | ~117K iterations (5 epochs on LSUN) | 5,850 iterations (15 epochs) |
| Noise prior | 100-dim uniform | 100-dim standard normal |
| Generator width | 1024 channels at 4×4 | Same (`feature_maps_g = 128`) |
| Feature extraction eval | CIFAR-10 and SVHN with an L2-SVM | Not included |
| Vector arithmetic | Face concepts (smiles, glasses) | Noise-averaging demo only |
| Guided backprop, "forgetting" windows | Included | Not included |
