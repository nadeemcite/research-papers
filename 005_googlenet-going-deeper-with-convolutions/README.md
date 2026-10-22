---
title: "005 GoogLeNet"
nav_order: 6
permalink: "/005_googlenet-going-deeper-with-convolutions/"
---
# GoogLeNet — Going Deeper with Convolutions
![GoogLeNet banner](../assets/banners/005.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/googlenet-going-deeper-with-convolutions)

- **Paper:** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- **Authors:** Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- **Lab, year:** Google, 2014 (CVPR 2015)
- **Video:** YouTube Short coming on 26 Oct 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

The paper introduces the *Inception* convolutional neural-network architecture, whose most famous incarnation, GoogLeNet, won the classification and detection tracks of the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14), with 6.67% top-5 error in classification.

The key idea is to keep the computational budget roughly constant while increasing both depth and width by replacing expensive, uniformly dense convolutions with *Inception modules*: parallel 1×1, 3×3 and 5×5 convolution branches whose outputs are concatenated, plus a parallel pooling branch.

Because stacking such naive modules would blow up the number of channels, the authors place 1×1 "dimension-reduction" convolutions before the 3×3 and 5×5 branches (and after the pooling branch). This bottlenecking, inspired by Network-in-Network and by Arora et al.'s theory of sparse deep representations, lets the network process visual information at multiple scales without a quadratic increase in computation (FLOPs, the number of arithmetic operations).

GoogLeNet is 22 layers deep when counting only layers with parameters (27 with pooling layers), uses ReLU everywhere, starts with a few conventional conv-pool layers, and adds two auxiliary classifiers (small side heads with a softmax, the layer that turns scores into class probabilities) at intermediate layers to help gradients reach the early layers during training.

## What Problem Does It Solve

Imagine you're looking at a photo and trying to figure out what's in it. Some objects are big and obvious (like a car filling the frame), some are small details (like a tiny bird in a tree), and some are medium-sized. If you only look through one magnifying glass, you'll miss things — the big magnifying glass can't see fine details, and the tiny one can't see the whole picture. You need multiple magnifying glasses of different sizes all at once.

Before GoogLeNet, most vision networks picked one filter size per layer (a single "magnifying glass"), so they could be good at seeing things at one scale but bad at others. GoogLeNet solves this with the **Inception module** — instead of choosing one filter size, it looks at the image through several different-sized filters at the same time (1×1, 3×3, 5×5) and combines what they all see.

It's like a Swiss Army knife that has a big blade, a small blade, and a scissors all built in — you don't have to pick just one tool. The clever trick is using tiny 1×1 filters as "funnels" to shrink the data before the bigger filters process it, so the whole thing stays fast and lightweight despite doing much more work. GoogLeNet was 22 layers deep but used 12× fewer parameters than AlexNet, the 2012 winner — proving you can be both smarter and lighter at the same time.

## Core Idea

- Replace the monolithic conv stack with a **repeating local block** that computes several filter sizes in parallel and concatenates them.
- Use **1×1 convolutions as bottlenecks** to reduce input depth before expensive 3×3 and 5×5 convolutions, and to project the pooling branch.
- Keep depth/width scaling **computationally efficient** so the whole network can be deep yet have fewer total parameters than earlier winners such as AlexNet.

## Key Method Details

- **Inception module (naive):** branches = 1×1 conv, 3×3 conv, 5×5 conv, 3×3 max-pool; outputs are concatenated along the channel axis.
- **Inception module with dimension reductions:** each 3×3 and 5×5 branch is preceded by a 1×1 reduction layer; the max-pool branch is followed by a 1×1 projection layer.
- **Stem:** conv 7×7/2 → max-pool 3×3/2 → conv 1×1 → conv 3×3 → max-pool 3×3/2.
- **Body:** 9 Inception modules arranged in stages (3a,3b / 4a–4e / 5a,5b) separated by stride-2 max-pooling.
- **Head:** global average pooling 7×7, dropout 40%, fully-connected classifier, softmax.
- **Auxiliary classifiers:** two side heads after Inception (4a) and (4d) during training, weighted by 0.3 and discarded at inference.
- **Input:** 224×224 RGB with mean subtraction; all convolutions use ReLU activation.

### Simplifications in This Notebook

- **CIFAR-10** (32×32 images, 10 classes) instead of ImageNet (224×224, 1,000 classes); 45,000 train, 5,000 validation and 10,000 test images
- A small "MiniGoogLeNet": a 3×3 conv stem and two Inception stages instead of 9 Inception modules
- **BatchNorm** after every conv (the paper did not use it), no auxiliary classifiers, no data augmentation
- Adam (learning rate 1e-3) for only 5 epochs, instead of the paper's SGD with momentum 0.9

### What the Notebook Result Shows

The MiniGoogLeNet has **304,122** parameters, while the plain 3×3 CNN baseline has **1,167,210** (3.84× more). After 5 epochs, the Inception model reached **77.0%** test accuracy and the plain CNN **76.5%**.

So the Inception network matched the plain CNN with about a quarter of the parameters. That is the paper's main point on a small scale. The 0.5-point accuracy gap is tiny and could change with another random seed.

## Influence

Inception/GoogLeNet showed that **network efficiency** (computational cost per unit accuracy) matters as much as raw capacity. Its multi-scale, multi-branch design influenced ResNeXt, DenseNet, NASNet, EfficientNet and modern vision backbones. The 1×1 bottleneck became a standard building block, appearing later in ResNet's bottleneck layers and as the 1×1 "pointwise" step in MobileNet's depthwise-separable convolutions.

### Follow-up and related papers

- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*. https://arxiv.org/abs/1512.00567
- Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2017). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. *AAAI*. https://arxiv.org/abs/1602.07261
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*. https://arxiv.org/abs/1512.03385
- Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. *CVPR*. https://arxiv.org/abs/1611.05431

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/005_googlenet-going-deeper-with-convolutions/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, Q&A (answers from the paper), citations, and common misconceptions about the paper |
