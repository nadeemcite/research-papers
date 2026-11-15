---
title: "021 VAE"
nav_order: 22
permalink: "/021_auto-encoding-variational-bayes-vae/"
---
# Auto-Encoding Variational Bayes (VAE)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/auto-encoding-variational-bayes-vae)

- **Paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **Authors:** Diederik P. Kingma, Max Welling
- **Lab, year:** University of Amsterdam, 2013 (ICLR 2014)
- **Video:** YouTube Short coming on 16 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

This paper introduced the **variational auto-encoder (VAE)**. A VAE is a pair of neural networks: an encoder that turns an image into a small set of hidden numbers, and a decoder that turns those numbers back into an image. Unlike a plain autoencoder, it treats the hidden numbers as random and pulls them toward a simple bell-curve distribution, so you can sample new numbers and decode brand-new images.

The hard part is training. The hidden numbers are sampled, and you cannot normally backpropagate through random sampling. The paper's **reparameterization trick** fixes this: it writes the sample as the encoder's mean plus its spread times fixed random noise, so gradients can flow through.

The result is a general recipe, which the authors call SGVB and AEVB, for training models with hidden variables on large datasets with ordinary stochastic gradient descent.

## What Problem Does It Solve

Imagine a big box of photos and a computer that should learn the "hidden ingredients" behind them, such as shape, thickness and slant for handwritten digits, without anyone labeling anything.

Working out those ingredients exactly is mathematically intractable for any interesting model. Older methods were too slow for big datasets or needed approximations that didn't scale.

The paper teaches a neural network to make a quick, approximate guess of the ingredients for each image, and trains that guesser together with the image generator. Everything is trained with the same gradient methods used for ordinary neural networks.

## Core Idea

- **Variational lower bound:** the model maximizes a lower bound on the log-likelihood of the data. It has two terms: a reconstruction term (how well the decoder rebuilds x from z) and a KL term that keeps the encoder's distribution close to the prior p(z) = N(0, I).
- **Reparameterization trick:** z = μ(x) + σ(x) ⊙ ε, with ε drawn from N(0, I). The randomness moves into ε, so the bound can be differentiated with respect to the encoder's parameters (Section 2.4).
- **Recognition model:** one encoder network approximates the posterior q(z\|x) for every data point, so inference for a new image is a single forward pass.

## Key Method Details

- **Distributions:** prior p(z) = N(0, I); encoder q(z\|x) = N(μ(x), diag σ²(x)); decoder p(x\|z) Bernoulli for binary data or Gaussian for real-valued data.
- **Closed-form KL:** for a Gaussian prior and posterior, the KL term is −½ Σ_j (1 + log σ_j² − μ_j² − σ_j²) (Appendix B).
- **Training:** minibatches of 100 with one sample per data point, optimized with Adagrad.
- **Experiments:** MNIST (500 hidden units) and Frey Face (200 hidden units). AEVB optimized the lower bound faster than the wake-sleep algorithm and reached a better solution in all experiments (Section 5).

### Simplifications in This Notebook

The notebook trains two VAEs on MNIST with one hidden layer of 512 ReLU units in the encoder and decoder: one with a 2-dimensional latent space for plotting and one with 20 dimensions. Each trains for 50 epochs with batch size 128 and Adam at learning rate 1e-3 (the paper used Adagrad and batches of 100). Pixels are used as values in [0, 1] with a Bernoulli (binary cross-entropy) likelihood.

### What the Notebook Result Shows

The 2D VAE's loss (negative ELBO per image) fell from **186.80** in epoch 1 to **144.38** in epoch 50, with a final reconstruction term of **137.90** and KL term of **6.48**. The 20D VAE reached a much lower loss, from **159.58** to **100.95**, because 20 latent numbers can carry far more detail.

The difference shows in the pictures: 20D reconstructions are sharp and keep each digit's identity, while the 2D model blurs them and turns some 4s and 5s into 9-like shapes. In the 2D latent map, 0, 1, 2, 6 and 7 form clear clusters while 4 and 9 overlap. Most 20D samples from the prior look like digits, though some are malformed.

## Influence

OpenAlex lists about 38,250 citations (38,253 when checked on 15 November 2026). [Rezende, Mohamed and Wierstra (2014)](https://arxiv.org/abs/1401.4082) developed the same reparameterization idea independently. Later work built on VAEs with discrete latents ([VQ-VAE](https://arxiv.org/abs/1711.00937)), disentanglement ([β-VAE](https://openreview.net/forum?id=Sy2fzU9gl)) and deep hierarchies ([NVAE](https://arxiv.org/abs/2007.03898), [Very Deep VAEs](https://arxiv.org/abs/2011.10650)). The original [DALL-E](https://arxiv.org/abs/2102.12092) used a discrete VAE (dVAE), from the same discrete-latent family as VQ-VAE. The authors later wrote [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) (2019).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/021_auto-encoding-variational-bayes-vae/solution.ipynb) | From-scratch PyTorch VAE on MNIST (2D and 20D latent spaces), with loss curves, reconstructions, the 2D latent map and samples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
