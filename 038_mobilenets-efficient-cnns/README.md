---
title: "038 MobileNets"
nav_order: 39
permalink: "/038_mobilenets-efficient-cnns/"
---
# MobileNets: Efficient CNNs for Mobile Vision
![MobileNets banner](../assets/banners/038.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mobilenets-efficient-cnns-for-mobile-vision)

- **Paper:** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- **Lab, year:** Google, 2017 (arXiv preprint)
- **Video:** YouTube Short coming on 28 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

MobileNets are a family of small, fast image-recognition networks built to run on phones and other small devices.

Their main building block is the depthwise separable convolution. It splits an ordinary convolution into two cheaper steps: first filter each input channel on its own, then mix the channels with a 1×1 convolution. With 3×3 filters this needs 8 to 9 times less computation.

Two simple knobs make the network smaller still: a width multiplier that thins every layer, and a resolution multiplier that shrinks the input image. The full MobileNet reaches 70.6% top-1 accuracy on ImageNet with 4.2 million parameters, close to VGG16's 71.5% while being 32 times smaller and needing 27 times less computation.

## What Problem Does It Solve

Imagine a recipe where every cook must taste every ingredient in every pot before adding anything. It works, but it is slow. A standard convolution is like that: each filter looks at every input channel and every nearby pixel at once.

A depthwise separable convolution splits the job. First each cook handles just one ingredient (one filter per channel). Then a quick final step mixes the results (the 1×1 convolution). The dish is almost as good, with far less work.

This matters because big image networks need a lot of memory and computation. On a phone they are slow and drain the battery, so photos often had to be sent to a server. MobileNets make networks small enough to run on the device, and let developers pick the size and speed they need.

## Core Idea

- **Depthwise convolution:** one D_K×D_K filter per input channel, costing D_K·D_K·M·D_F·D_F multiply-adds (Section 3.1).
- **Pointwise convolution:** a 1×1 convolution that combines the M channels into N outputs, costing M·N·D_F·D_F (Section 3.1).
- **Saving:** compared with a standard convolution's D_K·D_K·M·N·D_F·D_F, the cost falls by a factor of 1/N + 1/D_K², about 8 to 9 times for 3×3 filters (Section 3.1).
- **Width multiplier α:** thins every layer, so computation and parameters fall by roughly α² (Section 3.3).
- **Resolution multiplier ρ:** shrinks the input and every feature map, so computation falls by ρ²; the number of parameters stays the same (Section 3.4).

## Key Method Details

- **Architecture:** a standard 3×3 convolution first, then 13 depthwise separable blocks, average pooling, a fully connected layer and a softmax; 28 layers counting depthwise and pointwise layers separately. Every layer except the last is followed by batchnorm and ReLU (Section 3.2, Table 1, Figure 3).
- **Where the work goes:** 95% of the computation and 75% of the parameters are in the 1×1 convolutions, which run as fast matrix multiplies (Section 3.2, Table 2).
- **Training:** RMSprop with asynchronous gradient descent, less regularization and data augmentation than large models, and little or no weight decay on the depthwise filters (Section 3.2).
- **Separable vs full convolutions:** 70.6% vs 71.7% ImageNet accuracy, with 569 vs 4,866 million multiply-adds and 4.2 vs 29.3 million parameters (Table 4).
- **Thinner beats shallower:** at a similar budget, α = 0.75 reaches 68.4%, against 65.3% for a shallower MobileNet (Table 5).
- **The two knobs:** 70.6%, 68.4%, 63.7% and 50.6% for α = 1, 0.75, 0.5 and 0.25 (Table 6); 70.6%, 69.1%, 67.2% and 64.4% at 224, 192, 160 and 128 pixels (Table 7).
- **Other tasks:** Stanford Dogs 83.3% (Table 10), face attributes, geolocalization with PlaNet, and COCO object detection, where MobileNet SSD reaches 19.3% mAP (Table 13).

### Simplifications in This Notebook

The notebook builds the depthwise separable block from scratch in PyTorch and puts it into a small 5-layer network for CIFAR-10 (32×32 images, 10 classes). It compares this with a network of the same layout where every block is an ordinary 3×3 convolution. Both use width multiplier α = 1.0.

It counts parameters and multiply-adds, times one forward pass on a T4 GPU, and trains both networks the same way: Adam for 4 epochs on only 10,000 of the 50,000 training images. This is far shorter than real training.

### What the Notebook Result Shows

The MobileNet has **8.27×** fewer parameters (118,730 vs 981,802) and needs **8.19×** fewer multiply-adds, matching the paper's 8 to 9 times. But on the T4 GPU it was **slower**: in the saved run a forward pass took 1.194 ms against 0.703 ms for the standard network (the times vary between runs, but the MobileNet was slower in every run). Fewer multiply-adds did not mean faster here, because with one tiny image the GPU mostly waits on many small operations.

After 4 epochs, the MobileNet reached **49.38%** test accuracy and the standard network **55.36%**, a gap of about 6 points. Both are far from fully trained, so this short run shows the size saving clearly, but not the paper's small accuracy cost.

An earlier version of this notebook mixed numbers from different width multipliers and called the result a "dramatic efficiency gain" with "minimal accuracy loss", although its own timings showed the MobileNet was slower.

## Influence

OpenAlex lists about 34,300 citations ([34,322 when checked on 27 November 2026](https://openalex.org/W2612445135)). Google released the models as open source ([Google Research blog, June 2017](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/)), and they ship as a ready-made model in [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet). The follow-up [MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018) builds on its depthwise separable convolutions with inverted residual blocks.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/038_mobilenets-efficient-cnns/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
