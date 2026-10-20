---
title: "039 Wide ResNet"
nav_order: 40
permalink: "/039_wide-residual-networks/"
---
# Wide Residual Networks
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/wide-residual-networks)


**arXiv:** [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)

**Authors:** Sergey Zagoruyko, Nikos Komodakis (Université Paris-Est, École des Ponts ParisTech)

## Summary

Wide Residual Networks (WRNs) challenge the trend of making ResNets ever deeper and thinner. The authors point out that each fraction of a percent of extra accuracy from a very deep ResNet cost nearly a doubling of the number of layers. They also describe **diminishing feature reuse**: because gradients can flow through the identity shortcuts, many residual blocks in a very deep network learn very little, and only a few do the real work.

The core idea is to **widen** each residual block instead. A WRN-n-k has n layers in total, and each block has k times more feature channels than a standard ResNet block. The paper reports that a wide 16-layer network matches the accuracy of a 1000-layer thin ResNet with a comparable number of parameters, while being several times faster to train. On a Titan X, WRN-40-4 (about the same accuracy as ResNet-1001) is 8 times faster to train, and WRN-28-10 is 1.6 times faster and 0.92 points more accurate on CIFAR-10.

The authors also study **dropout between the two convolutions** of a residual block, not on the identity path. Earlier work (He et al., *Identity Mappings in Deep Residual Networks*) had put dropout on the identity path and found it hurt. In the paper, dropout inside the block gives gains on both thin and wide networks, most clearly on SVHN. WRN-28-10 with dropout reaches 3.89% error on CIFAR-10 (4.00% without), and WRN-16-8 with dropout reaches 1.54% error on SVHN, the best published result at the time.

## What problem does it solve

Imagine you're building a tower out of Lego blocks. Each block is a layer in a neural network. Researchers found that making the tower taller (more layers) made the network better, but only barely. Each time you doubled the height you got a tiny improvement, and the tower became wobbly: some blocks in the middle weren't doing anything useful anymore, because the signal could bypass them through a shortcut.

The Wide Residual Networks paper says: instead of building a super tall, skinny tower, build a **wider, shorter one**. Make each block fatter (more channels) rather than adding more blocks. It turns out a short, wide tower can be just as strong as the skyscraper, and it's much faster to build, because a GPU does one big wide operation faster than many small sequential ones. Think of hiring 10 skilled workers on each floor of a 5-story building instead of 1 worker on each floor of a 50-story building: the shorter building is easier to manage and gets the job done faster.

## Key method details

- **Block:** the pre-activation basic block `B(3,3)`: BN → ReLU → conv3×3 → (optional dropout) → BN → ReLU → conv3×3, plus a shortcut.
- **Layout:** an initial 3×3 conv with 16 channels, then three groups of `N = (n − 4) / 6` blocks with `16k`, `32k` and `64k` channels, then global average pooling and a linear layer.
- **Width is the only change:** with k = 1 this is a standard thin pre-activation ResNet.
- **Paper's training recipe:** SGD with Nesterov momentum 0.9, learning rate 0.1 dropped by 0.2 at epochs 60, 120 and 160, 200 epochs in total, weight decay 5e-4, batch size 128, dropout 0.3 on CIFAR.

## What the notebook result shows

The notebook trains seven WRNs on real CIFAR-10 with one identical recipe (20 epochs, one-cycle learning rate, one seed) on a single Kaggle T4 GPU. The recipe is much shorter than the paper's 200 epochs, so the numbers are lower than the paper's.

| Model | Parameters | Blocks | Test accuracy | Train time |
|---|---|---|---|---|
| WRN-16-1 | 0.18M | 6 | 88.66% | 60 s |
| WRN-28-1 | 0.37M | 12 | 89.70% | 96 s |
| WRN-40-1 | 0.56M | 18 | 89.71% | 136 s |
| WRN-52-1 | 0.76M | 24 | 90.31% | 175 s |
| WRN-16-2 | 0.69M | 6 | 91.90% | 93 s |
| WRN-16-4 | 2.75M | 6 | 93.07% | 205 s |
| WRN-16-8 | 10.96M | 6 | **94.18%** | 606 s |

- **Width beat depth at a similar budget.** WRN-16-2 (0.69M parameters) scored 2.19 points higher than WRN-40-1 (0.56M) and 1.59 points higher than WRN-52-1 (0.76M), and trained 1.5× and 1.9× faster. This is the paper's main claim in miniature.
- **Depth at width 1 helped only a little:** going from 6 to 24 blocks added under 2 points while training took about three times as long.
- **Lesion study:** in the 24-block thin network, 5 of 24 blocks could be switched off one at a time with under 1 point of accuracy loss (median drop 1.89 points). In the 6-block WRN-16-8 none could (median drop 78 points). This is consistent with diminishing feature reuse, but the networks differ in depth, so it is suggestive and not a controlled test.
- **Limits:** one seed per model, so differences of a few tenths of a point are noise, and a re-run can shift the numbers slightly. The notebook does not train a 1000-layer ResNet, so it does not test the paper's comparison against ResNet-1001, and it does not test dropout.

## Influence

Wide Residual Networks became one of the most cited ResNet variants, and later work on scaling networks, such as EfficientNet, treats width as one of the scaling dimensions. WRNs (especially WRN-28-10 and WRN-28-2) are widely used as baselines in semi-supervised learning, for example in MixMatch and FixMatch. The official code is at [github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks). The paper appeared at BMVC 2016.

---

## Resources

**Kaggle Notebook:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/wide-residual-networks)

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/039_wide-residual-networks/solution.ipynb) | From-scratch PyTorch WRN trained on real CIFAR-10 on a GPU: a width-vs-depth race over seven models, a lesion study, and prediction plots |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, interview Q&A, citations, and common misconceptions about the paper |
