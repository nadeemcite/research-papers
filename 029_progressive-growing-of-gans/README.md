---
title: "029 ProGAN"
nav_order: 30
permalink: "/029_progressive-growing-of-gans/"
---
# Progressive Growing of GANs for Improved Quality, Stability, and Variation
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/progressive-growing-of-gans)

- **Paper:** [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- **Authors:** Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
- **Lab, year:** NVIDIA, 2017 (ICLR 2018)
- **Video:** [Watch the YouTube Short](https://youtube.com/shorts/ds6J-BW-xqQ) on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

A GAN (generative adversarial network) trains two networks against each other: a generator that makes images, and a discriminator that tries to tell them from real ones. Before this paper, GANs struggled to make large, sharp images.

ProGAN starts both networks at a tiny 4×4 resolution and keeps adding layers that double it, up to 1024×1024. Each new layer is faded in gradually. The authors say this lets training first find the large-scale structure of the images, then move to finer detail.

The paper also adds three simple tricks: equalized learning rate, pixelwise feature normalization and a minibatch standard deviation layer. It produced 1024×1024 faces from CelebA-HQ, a high-quality version of CelebA that the authors built, and a record inception score of 8.80 on unsupervised CIFAR10.

## What Problem Does It Solve

Imagine learning to draw a face. You would not start with every eyelash. You would sketch the head's shape, then place the eyes and mouth, and only add fine detail at the end.

Earlier GANs tried to learn everything at once, the overall shape and the tiny details together. At high resolutions this was hard and often unstable.

ProGAN trains the way you would draw. The networks first learn rough 4×4 images, then sharper and sharper ones, building on what they already know at each step.

## Core Idea

- **Progressive growing:** the generator and discriminator mirror each other and grow together, one resolution step at a time. All existing layers stay trainable.
- **Fade-in:** when a new resolution is added, its output is blended with an upsampled copy of the old output, with the weight α rising from 0 to 1. This avoids sudden shocks to the layers that were already trained.
- **Minibatch standard deviation:** near the end of the discriminator, one extra feature map holds the average standard deviation across the batch. It helps the discriminator notice when generated images lack variety.

## Key Method Details

- **Equalized learning rate:** weights start from N(0, 1) and are scaled at runtime by He's per-layer constant, so every weight learns at the same speed under Adam (Section 4.1).
- **Pixelwise normalization:** after each 3×3 convolution in the generator, each pixel's feature vector is scaled to unit length (Section 4.2).
- **Loss:** WGAN-GP, with the generator and discriminator updated alternately and a small drift term (weight 0.001) that keeps the discriminator output near zero (Appendix A.1).
- **Training:** Adam with learning rate 0.001, β1 = 0 and β2 = 0.99. Each fade-in and each stable phase lasts 800k real images. Upsampling is nearest-neighbor, and downsampling is average pooling.

### Simplifications in This Notebook

The notebook grows a small ProGAN from 4×4 to 64×64 on MNIST digits, with 500 iterations per stage (200 fade-in, 300 stable). It uses the non-saturating GAN loss instead of WGAN-GP, a 128-dimensional latent instead of 512, and no running average of generator weights.

### What the Notebook Result Shows

Both networks grew from 4×4 to 64×64 with a fade-in at each step, and the samples improve with resolution. At 4×4 they are colored blocks, and at 8×8 grey blobs. Thin strokes appear at 16×16, and at 32×32 and 64×64 the samples are white digit-like strokes. A few look like a 0, 1, 7 or 9, but many are malformed.

The discriminator dominated early on. Over the last 100 stable iterations of the 4×4 stage, mean D loss was **0.051** and mean G loss **4.573**. By 64×64 the two were closer, at **0.772** and **1.998**. The generator loss spiked after each growth step, reaching **37.6** during the 8×8 stage.

The run measures no image-quality score and uses MNIST at 64×64, so it shows the mechanics of progressive growing, not the paper's face results.

## Influence

OpenAlex lists about 7,000 citations (7,024 when checked on 10 November 2026). The authors' [official code](https://github.com/tkarras/progressive_growing_of_gans) was released with the paper. NVIDIA's [StyleGAN](https://arxiv.org/abs/1812.04948) built on this work, and [StyleGAN2](https://arxiv.org/abs/1912.04958) later revisited progressive growing and replaced it with a design that does without it.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/029_progressive-growing-of-gans/solution.ipynb) | From-scratch PyTorch implementation of a small progressive GAN on MNIST, with sample grids and loss curves |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
