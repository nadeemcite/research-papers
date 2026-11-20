---
title: "084 DDPM"
nav_order: 85
permalink: "/084_denoising-diffusion-probabilistic-models/"
---
# Denoising Diffusion Probabilistic Models (DDPM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/denoising-diffusion-probabilistic-models-ddpm)

- **Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Lab, year:** UC Berkeley, 2020 (NeurIPS 2020)
- **Video:** YouTube Short coming on 20 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

A diffusion model makes images by learning to undo noise. During training, it takes real images, adds a random amount of Gaussian noise, and learns to predict exactly which noise was added.

To make a new image, it starts from pure noise and removes a little predicted noise at a time, over 1000 steps, until a clean image appears.

This paper made diffusion models work well. It trains the network to predict the noise with a simple mean-squared-error loss. On CIFAR10 it reached an Inception score of 9.46 and an FID of 3.17, a state-of-the-art FID at the time.

## What Problem Does It Solve

Imagine slowly adding TV static to a photo until only static is left. Going forward is easy. Going backwards, turning static into a photo, looks impossible.

But if you only have to remove a tiny bit of static at a time, each step is a manageable guess. DDPM trains a network to make that small guess, then chains 1000 of them together.

Before this, the best image generators were mostly GANs, which pit two networks against each other and can be hard to train. DDPM trains with plain regression on the noise and still produced samples that matched or beat GANs on CIFAR10.

## Core Idea

- **Forward process:** add Gaussian noise over T steps with a fixed schedule β_t. Any noisy version can be made in one jump: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε, where ᾱ_t is the product of (1 − β_s) up to step t.
- **Reverse process:** a network learns to go back one step at a time. Each step is a Gaussian whose variance is fixed, not predicted: σ_t² = β_t, or the posterior variance β̃_t, which the authors found gave similar results (Section 3.2).
- **Predict the noise:** the network ε_θ(x_t, t) predicts the noise ε, and the step's mean is μ = (1/√α_t) · (x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t)), with α_t = 1 − β_t (Section 3.2).
- **Simple loss:** train with L_simple = E‖ε − ε_θ(x_t, t)‖² at a random timestep, an unweighted version of the variational bound (Section 3.4).

## Key Method Details

- **Schedule:** T = 1000, β_t rising linearly from 10⁻⁴ to 0.02 (Section 4).
- **Network:** a U-Net like an unmasked PixelCNN++, with group normalization, sinusoidal timestep embeddings and self-attention at the 16×16 resolution; the CIFAR10 model has 35.7M parameters (Section 4, Appendix B).
- **Training:** Adam with learning rate 2 × 10⁻⁴, batch 128 on CIFAR10, dropout 0.1, and an exponential moving average (EMA) of the weights with decay 0.9999 (Appendix B).
- **Results:** CIFAR10 Inception score 9.46 and FID 3.17 (Table 1). In the ablation, predicting the noise with L_simple beat predicting the mean, and learning the variance made training unstable (Section 4.2, Table 2).

### Simplifications in This Notebook

The notebook implements DDPM from scratch in PyTorch on FashionMNIST, resized to 32×32 grayscale. It keeps the paper's T = 1000, β schedule, learning rate, batch size and simple loss, and samples with the EMA model through all 1000 steps, using the posterior variance β̃_t. The U-Net is much smaller (1.95M parameters, channels 32-64-128, attention at 16×16 only), has no dropout, and trains for 25 epochs (11,700 steps) with EMA decay 0.999.

### What the Notebook Result Shows

Training worked: the noise-prediction loss fell from **0.0849** in the first epoch to **0.0257** in the last. Most of the 64 samples are clearly recognisable clothing: shirts, trousers, dresses, sneakers, sandals, boots and bags. Some are blurry or distorted.

Simple pixel statistics match the real images (mean −0.466 vs −0.425, 52.3% vs 50.4% background pixels), but we did not measure FID. An earlier version of this notebook produced pure noise because of an attention bug; it also trained for only 3 epochs on 5,000 images with T = 100.

## Influence

OpenAlex lists about 20,800 citations ([20,776 when checked on 19 November 2026](https://openalex.org/W3036167779)). Later work built directly on it: [DDIM](https://arxiv.org/abs/2010.02502) sped up sampling, [Latent Diffusion](https://arxiv.org/abs/2112.10752) ran diffusion in a compressed latent space, and [DALL-E 2](https://arxiv.org/abs/2204.06125) used diffusion models to generate images from text.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/084_denoising-diffusion-probabilistic-models/solution.ipynb) | From-scratch PyTorch DDPM: forward noising, a small U-Net with 16×16 attention, EMA training on FashionMNIST and 1000-step sampling |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
