---
title: "004 VGGNet"
nav_order: 5
permalink: "/004_vggnet-very-deep-convolutional-networks/"
---
# VGGNet — Very Deep Convolutional Networks for Large-Scale Image Recognition
![VGGNet banner](../assets/banners/004.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/vggnet-very-deep-convolutional-networks)

- **Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Authors:** Karen Simonyan, Andrew Zisserman
- **Lab, year:** Visual Geometry Group, University of Oxford, 2014 (ICLR 2015)
- **Video:** YouTube Short coming on 25 Oct 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

VGGNet is a careful study of how the **depth** of a convolutional network affects its accuracy on large-scale image recognition. Simonyan and Zisserman kept the rest of the design fixed and used only very small 3×3 convolution filters throughout, then increased depth step by step from 11 to 19 weight layers. Accuracy improved significantly with depth.

The deepest configurations, now known as **VGG-16** and **VGG-19**, set the template for a simple, uniform ConvNet: stacks of 3×3 convolutions with stride 1, 2×2 max-pooling between stages, the number of channels doubling after each pool, and three fully connected layers at the end. The team placed 1st in localization and 2nd in classification at ILSVRC 2014, and the released models transferred well to other datasets, which made VGG one of the most widely used backbones (the base network that other models are built on) and feature extractors in computer vision.

## What Problem Does It Solve

Imagine you're building a tower out of blocks. You discover that using small, identical blocks stacked one on top of another makes a taller and sturdier tower than using a few big, fancy blocks. Before VGGNet, people building AI vision systems were using big, complicated blocks (large filters like 7×7 or 11×11) and wondering why their towers couldn't go very high without becoming unstable.

VGGNet solves this by showing that stacking many small, simple 3×3 blocks — the same size block, over and over — lets you build a much deeper and smarter network than using a few large, complex blocks. It's like building with tiny Lego bricks instead of big chunks: you get more flexibility, more places to connect, and the whole structure is sturdier. Three small blocks stacked together can "see" the same area as one big block, but with fewer pieces and more flexibility. This simple idea — go deep with small, uniform blocks — became the recipe for almost every image-recognition AI that came after.

## Core Idea

Replace large receptive-field filters (for example a single 7×7 or 11×11) with a **stack of 3×3 convolutions**. Three stacked 3×3 layers have the same effective receptive field as one 7×7 layer, but with **fewer parameters** and **more non-linearity** (three ReLUs instead of one), which makes the decision function more expressive. Beyond that insight, VGGNet is strikingly uniform: every convolution uses 3×3 filters with stride 1 and same padding, every downsampling is done by a 2×2 max-pool, and the number of filters doubles after each pooling.

## Key Method Details

A typical VGG-style block:

```
Conv3-64 → Conv3-64 → MaxPool
Conv3-128 → Conv3-128 → MaxPool
Conv3-256 → Conv3-256 → Conv3-256 → MaxPool
...
```

- **3×3 convolutions, stride 1, padding 1** keep spatial resolution constant inside each block.
- **ReLU** activation after every convolution.
- **2×2 max-pooling, stride 2** halves spatial dimensions between blocks.
- **Channels double after each pool**: 64 → 128 → 256 → 512 → 512.
- **Classifier head**: flattened features → FC-4096 → ReLU → Dropout → FC-4096 → ReLU → Dropout → FC-1000 → Softmax.
- VGG-16 = 13 convolutional + 3 fully connected weight layers. VGG-19 = 16 convolutional + 3 FC.

### ILSVRC 2014 results (from the paper)

| Entry | Top-5 test error |
|---|---|
| VGG (single net, multi-crop/dense eval.) | 7.0% |
| VGG (2-net ensemble) | 6.8% |
| VGG (7-net ensemble, original submission) | 7.3% |
| GoogLeNet (winner, 7-net ensemble) | 6.7% |

Official VGG page: https://robots.ox.ac.uk/~vgg/research/very_deep/

### Simplifications in This Notebook

- **CIFAR-10** (32×32 images, 10 classes) instead of ImageNet (224×224 crops, 1,000 classes)
- A VGG-D-style stack of 13 3×3 conv layers, but with a small head (FC-512 → FC-512 → FC-10) instead of FC-4096 → FC-4096 → FC-1000; about 15.3M parameters
- **BatchNorm** after every conv layer (the paper did not use it)
- SGD with momentum 0.9, weight decay 5e-4 and learning rate 0.01 as in the paper, but a cosine schedule for only 20 epochs (the paper trained for 74 epochs)
- One network, no multi-scale training and no multi-crop test

### What the Notebook Result Shows

**The training in the saved run failed.** From epoch 2 to epoch 20, the train loss stayed at about 2.3026 (the value for random guessing over 10 classes, ln 10). The test accuracy was exactly **10.00%** in every epoch, which is chance level. So this run does **not** show VGGNet learning; it needs a code fix and a re-run.

The parts that do not need training still work. Three stacked 3×3 convs with 64 channels use 110,592 weights, while one 7×7 conv uses 200,704 (44.9% fewer), and the stack has 3 ReLUs instead of 1. The "plain 5×5 CNN" in the notebook has fewer parameters (10.87M vs 15.25M), but only because it has 5 conv layers and a single small classifier layer, so it is not a like-for-like test.

## Influence

VGGNet placed **1st in localization** and **2nd in classification** at ILSVRC 2014, but its real impact was proving that a simple, deep, uniform design could rival hand-crafted complexity. It became a go-to backbone for transfer learning, object detection (Faster R-CNN), segmentation (FCN), and style transfer. Its 3×3 stacking philosophy influenced ResNet, DenseNet, and nearly every ConvNet that followed.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/004_vggnet-very-deep-convolutional-networks/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), and common misconceptions about the paper |
