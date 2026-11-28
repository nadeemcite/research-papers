---
title: "049 Xception"
nav_order: 50
permalink: "/049_xception-depthwise-separable-convolutions/"
---
# Xception: Deep Learning with Depthwise Separable Convolutions
![Xception banner](../assets/banners/049.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/xception-depthwise-separable-convolutions)

- **Paper:** [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- **Authors:** François Chollet
- **Lab, year:** Google, 2017 ([CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html))
- **Video:** YouTube Short coming on 29 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

Xception is an image-recognition network built almost entirely from depthwise separable convolutions. Its name means "Extreme Inception".

The paper starts from a new way of looking at Inception modules. An Inception module first mixes the channels with 1×1 convolutions into 3 or 4 groups, then looks at nearby pixels inside each group. If you keep adding groups until every channel is its own group, you get a depthwise separable convolution: one spatial filter per channel, then a 1×1 convolution that mixes the channels.

Xception replaces every Inception module with these convolutions and adds residual connections. It has 36 convolutional layers in 14 modules and almost the same number of parameters as Inception V3 (22.9 vs 23.6 million).

On ImageNet it is slightly better than Inception V3 (79.0% vs 78.2% top-1). On Google's much larger JFT dataset (350 million images, 17,000 classes) the gain is bigger: 4.3% relative. Because the two networks are the same size, the paper concludes that Xception uses its parameters more efficiently.

## What Problem Does It Solve

Think of reading a page of colour-coded notes. One way is to read every colour and every word at the same time. A simpler way is to first read each colour on its own, then combine what you found. A regular convolution does the first: each filter looks at all channels and all nearby pixels at once.

Inception modules were a step toward the simpler way: they split the channels into a few groups and handle each group separately. Xception asks what happens if you go all the way, with one group per channel.

The answer is a network that is easy to define (a plain stack of layers with shortcuts, rather than Inception's many branches) and that gets better results than Inception V3 at the same size.

## Core Idea

- **The spectrum:** a regular convolution is one channel segment; Inception modules split a few hundred channels into 3 or 4 segments; a depthwise separable convolution has one segment per channel (Section 1.2).
- **Depthwise separable convolution:** a depthwise convolution (one spatial filter per channel) followed by a pointwise 1×1 convolution that projects onto a new set of channels (Section 1.2).
- **Two small differences from an "extreme" Inception module:** the order (depthwise first, then 1×1, which the paper argues does not matter in a stack) and no non-linearity between the two steps (Section 1.2).
- **The Xception hypothesis:** cross-channel and spatial correlations can be mapped completely separately (Section 3).

## Key Method Details

- **Architecture:** an entry flow, a middle flow repeated 8 times, and an exit flow; 36 convolutional layers in 14 modules, all with linear residual connections except the first and last. Every convolution is followed by batch normalization, and separable convolutions use a depth multiplier of 1 (Section 3, Figure 5).
- **Training on ImageNet:** SGD with momentum 0.9, initial learning rate 0.045, decayed by 0.94 every 2 epochs. On JFT: RMSprop, learning rate 0.001. These settings were tuned for Inception V3 and used unchanged for Xception (Section 4.2).
- **Regularization:** weight decay 1e-5 (Inception V3 used 4e-5), dropout 0.5 before the classifier on ImageNet, no auxiliary tower; Polyak averaging at test time (Sections 4.2 and 4.3).
- **ImageNet:** 79.0% top-1 and 94.5% top-5, against 78.2% and 94.1% for Inception V3 (Table 1).
- **JFT:** MAP@100 of 6.70 vs 6.36 without fully connected layers, and 6.78 vs 6.50 with them (Table 2).
- **Size and speed:** 22,855,952 vs 23,626,728 parameters, and 28 vs 31 training steps per second, so Xception is marginally slower (Table 3).
- **Ablations:** without residual connections Xception converges more slowly and to a worse result (Section 4.6, Figure 9); adding a ReLU or ELU between the depthwise and pointwise steps makes it worse (Section 4.7, Figure 10).

### Simplifications in This Notebook

The notebook builds the depthwise separable convolution and the Xception module (two separable convolutions plus a residual connection) from scratch in PyTorch. It puts them into a small network for CIFAR-10 (32×32 images, 10 classes): a first convolution, then 5 modules with up to 128 channels, instead of the paper's 14 modules.

For comparison, it builds an Inception-style network with the same layout and channel counts, where every module runs 1×1, 3×3 and 5×5 convolutions side by side. These are full convolutions, so this network has about 3.7 times as many parameters. The two are not matched in size, unlike the paper's Xception and Inception V3.

Both are trained the same way: SGD for only 3 epochs on the 50,000 training images. This is far shorter than real training.

### What the Notebook Result Shows

The small Xception has **97,422** parameters and the Inception-style network **363,720**, 3.73 times as many.

After 3 epochs, the small Xception reached **64.80%** test accuracy and the Inception-style network **67.22%**. So in this short run the Xception was smaller but **less accurate**, by 2.42 points. Both were still improving when training stopped, so this does not show the paper's result.

An earlier version of this notebook said the two networks had the same parameter budget and trained for 15 epochs, although its own output showed 3.73 times as many parameters and 3 epochs.

## Influence

OpenAlex lists about 24,400 citations ([24,405 when checked on 28 November 2026](https://openalex.org/W2531409750)). An open-source implementation ships in [Keras Applications](https://keras.io/api/applications/xception/), as the paper itself notes (Section 3), and in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception). The [MobileNets](https://arxiv.org/abs/1704.04861) paper cites it as showing how to scale up depthwise separable filters to beat Inception V3 (its Section 2), and [MobileNetV2](https://arxiv.org/abs/1801.04381) cites it among the efficient architectures built on depthwise separable convolutions (its Section 3.1).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/049_xception-depthwise-separable-convolutions/solution.ipynb) | From-scratch PyTorch implementation of the Xception module, a small Xception-style network and an Inception-style comparison |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
