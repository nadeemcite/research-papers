---
title: "VAE: Code architecture"
parent: "021 VAE"
nav_order: 1
---
# Code Architecture — Auto-Encoding Variational Bayes (VAE)

## Overview

The notebook implements a **Variational Auto-Encoder** from scratch in PyTorch, following the original paper by Kingma & Welling (2013). The model is an encoder-decoder architecture with the reparameterization trick, trained on MNIST to minimize the negative ELBO (reconstruction loss + KL-divergence). After training, the 2D latent space is visualized with scatter plots and latent-space interpolations.

## Notebook Structure

### Section 1: Setup & Imports
- Imports torch, torchvision, numpy and matplotlib; seeds `random`, NumPy and PyTorch with 42.
- Deterministic cuDNN/cuBLAS settings, so a re-run on the same hardware reproduces the saved numbers.
- Figures shown as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`.
- Device selection (`cuda` if available, else `cpu`).

### Section 2: Data Loading — MNIST
- Loads MNIST via `torchvision.datasets.MNIST`, downloaded quietly to a temp folder.
- Applies standard preprocessing: flatten to 784-dim vector, normalize to [0, 1].
- Creates `DataLoader` with minibatch size M=128 (paper used 100; 128 is a standard GPU-friendly batch size).
- **Data flow:** Each image is a 28x28 grayscale image → flattened to a 784-dim tensor.
- **Shapes:** Batch: `(B, 784)` where B is the minibatch size.

### Section 3: Encoder Network (Probabilistic Encoder)
- **Class:** `Encoder(nn.Module)`
- An MLP that maps input x (784-dim) to the parameters of the approximate posterior q(z\|x).
- Architecture: 784 → 512 (hidden, ReLU) → two parallel linear layers:
  - `fc_mu`: 512 → latent_dim (mean of q(z\|x))
  - `fc_logvar`: 512 → latent_dim (log-variance of q(z\|x))
- Outputs: `mu` (B, latent_dim) and `logvar` (B, latent_dim).
- The use of log-variance (instead of variance directly) ensures numerical stability (variance is always positive via exp(logvar)).
- **Shapes:** Input `(B, 784)` → hidden `(B, 512)` → mu `(B, latent_dim)`, logvar `(B, latent_dim)`.

### Section 4: Reparameterization Trick
- **Function:** `reparameterize(mu, logvar)`
- Implements z = mu + sigma * epsilon, where:
  - `std = exp(0.5 * logvar)` (standard deviation from log-variance)
  - `eps = torch.randn_like(std)` (auxiliary noise from standard normal)
  - `z = mu + std * eps` (reparameterized sample)
- This is the key innovation: the sampling is now a deterministic function of (mu, sigma, epsilon), so gradients can flow through mu and sigma via backpropagation.
- **Shapes:** Input mu `(B, latent_dim)`, logvar `(B, latent_dim)` → output z `(B, latent_dim)`.

### Section 5: Decoder Network (Probabilistic Decoder)
- **Class:** `Decoder(nn.Module)`
- An MLP that maps latent z to the parameters of the likelihood p(x\|z).
- Architecture: latent_dim → 512 (hidden, ReLU) → 784 (output, Sigmoid).
- The sigmoid output maps to [0, 1], suitable for Bernoulli likelihood on normalized MNIST pixels.
- **Shapes:** Input `(B, latent_dim)` → hidden `(B, 512)` → output `(B, 784)`.

### Section 6: VAE Model
- **Class:** `VAE(nn.Module)`
- Combines Encoder + reparameterization + Decoder into a single module.
- **Forward pass:** `x → encode → (mu, logvar) → reparameterize → z → decode → x_recon`
- Returns: `x_recon` (reconstructed image), `mu`, `logvar` (for loss computation).
- **Shapes:** Input `(B, 784)` → `x_recon (B, 784)`, `mu (B, latent_dim)`, `logvar (B, latent_dim)`.

### Section 7: Loss Function — Negative ELBO
- **Function:** `vae_loss(x_recon, x, mu, logvar)`
- The ELBO has two terms:
  1. **Reconstruction loss (BCE):** Binary Cross-Entropy between x and x_recon. This is the expected negative log-likelihood: -E[log p(x\|z)]. Computed as `F.binary_cross_entropy(x_recon, x, reduction='sum')`.
  2. **KL-divergence:** Analytical closed-form for Gaussian prior N(0,I) and Gaussian posterior: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`. This regularizes the encoder's posterior to stay close to the prior.
- Total loss = reconstruction_loss + KL_divergence (both summed over the batch, then divided by batch size for per-sample average).
- **Note:** The paper formulates this as maximizing the ELBO; we minimize the negative ELBO.

### Section 8: Training Loop
- Optimizer: Adam (lr=1e-3), following common VAE training practice (paper used Adagrad).
- Epochs: 50 for each model (the 2D and the 20D VAE), batch size 128.
- Per-epoch: iterate over minibatches, forward pass, compute loss, backward, optimizer step.
- Tracks and prints: total loss, reconstruction loss, KL divergence per epoch.
- Saves loss curves for visualization.
- **Data flow:** `(x, _) → model(x) → (x_recon, mu, logvar) → loss → backward → step`

### Section 9: Training Loss Curves
- Plots the total loss (negative ELBO), the reconstruction term (BCE) and the KL term per epoch for the 2D model.

### Section 10: Visualization — Original vs. Reconstructed
- Shows 10 test images and the 2D model's reconstructions side by side.

### Section 11: Visualization — 2D Latent Space Scatter
- Encodes all 10,000 test images to their means (mu_1, mu_2) and plots them colored by digit class.

### Section 12: Visualization — Latent Space Interpolation
- Decodes a 20 × 20 grid of points from −2 to 2 in each latent dimension, showing the learned 2D manifold.

### Section 13: Visualization — Linear Interpolation Between Digits
- Encodes a test 7 and a test 1, linearly interpolates between their latent means and decodes 10 steps.

### Section 14: Random Sampling from the Prior
- Samples 64 points from N(0, I) and decodes them with the 2D model.

### Section 15: Higher-Dimensional Latent Space (20D)
- Trains a second VAE with a 20-dimensional latent space, same architecture and settings, for 50 epochs.
- Compares 2D and 20D reconstructions of the same 10 test images, and decodes 64 samples from the 20D prior.

### Section 16: Summary
- `### What the run above shows`: the saved run's loss numbers and what the plots show.

## Deliberate Simplifications vs. Full Paper

1. **Single hidden layer MLPs:** We follow the same one-hidden-layer design (512 units instead of 500) but use ReLU where the paper uses tanh (Appendix C).

2. **Bernoulli likelihood only:** We use Bernoulli (BCE) for MNIST as the paper does. The paper also tested Gaussian likelihood on the Frey Face dataset; we omit this for simplicity.

3. **Adam instead of Adagrad:** The paper used Adagrad. We use Adam (the modern standard for VAEs) for faster convergence. The learning rate (1e-3) follows common practice.

4. **Fixed latent dimensionality:** We primarily use latent_dim=2 for visualization (matching the paper's 2D manifold plots). We also include a higher-dimensional variant (latent_dim=20) for better reconstruction quality.

5. **No weight decay:** The paper added a small weight decay term (Gaussian prior on parameters). We omit this as it has minimal effect on MNIST with modern optimizers.

6. **L=1 sample per datapoint:** The paper found L=1 sufficient with large minibatches. We use L=1 (single reparameterized sample per forward pass), which is the standard VAE practice.

7. **No comparison with wake-sleep:** The paper compared AEVB to the wake-sleep algorithm. We omit this comparison and focus on the VAE itself.
