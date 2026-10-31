---
title: "024 Conditional GAN"
nav_order: 25
permalink: "/024_conditional-generative-adversarial-nets/"
---
# Conditional Generative Adversarial Nets (CGAN)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/conditional-generative-adversarial-nets)

- **Paper:** [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) (arXiv:1411.1784)
- **Authors:** Mehdi Mirza, Simon Osindero
- **Lab, year:** Université de Montréal & Flickr / Yahoo, 2014
- **Video:** YouTube Short coming on 1 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

Conditional Generative Adversarial Nets (CGANs) extend the original GAN by giving extra information, most often a class label, to both the generator and the discriminator. In a standard GAN, the generator turns random noise into data samples, but there is no way to choose *which* kind of sample it makes.

The conditional GAN fixes this. It joins (concatenates) a one-hot class label to the noise vector going into the generator, and to the image going into the discriminator. The model then learns to generate samples of the class you ask for. The minimax objective becomes `min_G max_D V(D,G) = E[log D(x\|y)] + E[log(1 - D(G(z\|y)))]`, where y is the conditioning label. The paper shows this on MNIST (making specific digits on demand) and on the MIR Flickr 25,000 dataset for multi-modal image tagging.

CGANs became the base for a long line of conditional generation work: Pix2Pix (image-to-image translation), CycleGAN (unpaired translation), conditional image synthesis, and text-to-image generation (StackGAN, AttnGAN). The same idea of feeding the condition into the model shows up again in conditional diffusion models (classifier-free guidance).

## What Problem Does It Solve

Imagine you have a robot that can **draw** pictures of handwritten digits — but you have no way to tell it *which* digit to draw. You press the button and it might give you a 3, or a 7, or a 9 — it's random. That's what a regular GAN does: it generates images, but you can't control what comes out.

Now imagine you want to order a specific meal at a restaurant. If the kitchen just sends out random dishes, that's frustrating — you want to say "I'll have the pasta" and get pasta. The conditional GAN is like giving the kitchen your order: you say "give me a picture of the digit 5," and the generator produces a 5. It does this by showing both the generator and the discriminator the class label (the "order") during training. The generator learns "when someone asks for a 5, here's what a 5 looks like." The discriminator learns "not only must this image look real, it must also look like the digit it claims to be."

This simple idea — feeding the label into both networks — unlocked controlled generation across all of AI: generating specific faces, translating sketches to photos, turning text descriptions into images, and much more.

## Core Idea

### The Conditional Minimax Game

The GAN objective is extended by conditioning on auxiliary information y:

```
min_G  max_D  V(D, G) = E_{x~p_data}[log D(x|y)] + E_{z~p_z}[log(1 - D(G(z|y)))]
```

- **G(z\|y)**: generator takes noise z AND label y, produces a class-conditioned sample
- **D(x\|y)**: discriminator takes image x AND label y, decides if the image is real AND matches the label
- **y**: any auxiliary information (class label, text embedding, data from another modality)

### Conditioning Mechanism

- **Generator**: noise vector z (dim 100) and one-hot label y (dim 10 for MNIST) are concatenated and fed into a joint hidden layer. The generator must produce an image that both looks real AND matches the given label.
- **Discriminator**: image x (dim 784) and one-hot label y (dim 10) are concatenated and fed into the discriminator. The discriminator must verify that the image is real AND that it matches the claimed label.

## Key Method Details

### Architecture (from the paper)

- **Generator**: noise z (100-dim, uniform on the unit hypercube) → ReLU layer of 200 units, and label y → ReLU layer of 1000 units. The two are joined in a combined ReLU layer of 1200 units, followed by a sigmoid output layer of 784 units (one per MNIST pixel).
- **Discriminator**: x → Maxout(240, 5 pieces) + y → Maxout(50, 5 pieces) → joint Maxout(240, 4 pieces) → Sigmoid(1).
- **Training**: SGD with momentum (0.5→0.7), exponential learning rate decay (0.1→0.000001), dropout 0.5, batch size 100.

### Our Implementation

We simplify the architecture to plain MLPs that concatenate z and y at the input (rather than separate per-input hidden layers): the generator is 110 → 256 → 512 → 784 with LeakyReLU and a Tanh output (images scaled to [-1, 1]), and the discriminator is 794 → 512 → 256 → 1 with LeakyReLU, dropout 0.3 and a sigmoid output. Noise is drawn from a standard normal, and training uses Adam (lr 0.0002, β₁ = 0.5, batch 64) instead of SGD with momentum. The conditioning mechanism (feeding the one-hot label to both networks) is the paper's.

### What the Notebook Result Shows

The notebook trains the simplified MLP CGAN on MNIST for 100 epochs on a Tesla T4. The generator has 562,192 parameters and the discriminator 538,625. The losses settle by about epoch 50 and end at **D 1.287 / G 0.857** (final-epoch averages). That is close to, but not exactly at, the balanced values of 1.386 and 0.693, so the discriminator keeps a small edge.

The key result is the class grid: **every row shows the digit it was asked for**. Styles vary across the noise samples, and only a few samples are malformed (such as a 2 that looks closer to a 3). Fixing one noise vector and changing only the label gives ten recognisable digits 0 to 9. So the label controls the class, and the noise controls the style.

The paper reports a Parzen-window log-likelihood (a way to score how likely real test images are under the generator) instead of sample grids. This notebook does not compute it.

## Influence

- **Direct descendants**: Pix2Pix (2016), CycleGAN (2017), StackGAN (2017), AttnGAN (2018), BigGAN (2019), conditional diffusion models (classifier-free guidance, 2022)
- **Applications**: text-to-image generation, image-to-image translation, class-conditional image synthesis, data augmentation with specific class targets, controllable generation
- **Key insight**: The conditioning principle — feeding labels into both generator and discriminator — became the universal mechanism for controlled generation across all generative frameworks, not just GANs
- **Multi-modal learning**: The paper's second experiment (image tagging on MIR Flickr) foreshadowed multi-modal AI, connecting image features to word embeddings via adversarial training

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/024_conditional-generative-adversarial-nets/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, Q&A (answers from the paper), citations, and common misconceptions about the paper |
