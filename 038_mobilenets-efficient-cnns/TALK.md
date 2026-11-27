---
title: "MobileNets: Talk & Q&A"
parent: "038 MobileNets"
nav_order: 2
---
# TALK.md — MobileNets

## Press & Blog Coverage

1. [**MobileNets: Open-Source Models for Efficient On-Device Vision**](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/), Google Research blog, 14 June 2017. Google's announcement of the open-source release of MobileNets for TensorFlow, with pretrained models of different sizes.

2. [**tf.keras.applications.mobilenet**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet), TensorFlow documentation. A ready-made MobileNet in Keras, with ImageNet weights.

3. [**MobileNets on Hugging Face Papers**](https://huggingface.co/papers/1704.04861), the paper's page on Hugging Face (the old Papers With Code link now redirects here).

4. [**MobileNet_v1 in the TensorFlow models repository**](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), GitHub. The pretrained MobileNet v1 checkpoints for the different width multipliers and input resolutions.

5. [**MobileNetV2: The Next Generation of On-Device Computer Vision Networks**](https://research.google/blog/mobilenetv2-the-next-generation-of-on-device-computer-vision-networks/), Google Research blog. The follow-up model, which builds on MobileNetV1's depthwise separable convolutions.

## Q&A (answers from the paper)

**Q1: Why split a standard convolution into a depthwise and a pointwise convolution?**
A: A standard convolution filters and combines its inputs in one step. The depthwise separable convolution splits this into one layer for filtering (one 3×3 filter per input channel) and one 1×1 layer for combining the channels. The cost falls from D_K·D_K·M·N·D_F·D_F to D_K·D_K·M·D_F·D_F + M·N·D_F·D_F, so with 3×3 kernels it needs 8 to 9 times less computation (Section 3.1). For an internal layer with 512 input and 512 output channels on a 14×14 map, this cuts 462 million multiply-adds and 2.36 million parameters to 52.3 million and 0.27 million (Table 3).

**Q2: Why not just make a standard network smaller?**
A: The paper compares the two directly. Full convolutions instead of depthwise separable ones give 71.7% ImageNet accuracy against 70.6%, at 4,866 vs 569 million multiply-adds and 29.3 vs 4.2 million parameters (Section 4.1, Table 4). At a similar budget, a thinner MobileNet (α = 0.75, 68.4%) is about 3 points better than a shallower one with 5 layers removed (65.3%) (Section 4.1, Table 5).

**Q3: What do the width multiplier and the resolution multiplier do?**
A: The width multiplier α thins every layer, reducing computation and parameters roughly by α² (Section 3.3). The resolution multiplier ρ shrinks the input image and every feature map, reducing computation by ρ² but not the number of parameters (Section 3.4). Accuracy drops off smoothly: 70.6%, 68.4%, 63.7% and 50.6% for α = 1, 0.75, 0.5 and 0.25 (Table 6), and 70.6% to 64.4% as the input shrinks from 224 to 128 pixels (Table 7).

**Q4: Which nonlinearity does MobileNet use?**
A: In the paper, every layer is followed by batchnorm and a ReLU, except the final fully connected layer, which feeds a softmax (Section 3.2, Figure 3). The paper does not mention ReLU6. ReLU6 appears in the released TensorFlow code ([`mobilenet_v1.py`](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) uses `tf.nn.relu6`), and the [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) explains it as being robust with low-precision computation (its Section 4).

**Q5: How does it compare with SqueezeNet?**
A: MobileNet with α = 0.5 at 160×160 pixels reaches 60.2% on ImageNet with 1.32 million parameters and 76 million multiply-adds. SqueezeNet reaches 57.5% with 1.25 million parameters and 1,700 million multiply-adds (Table 9). That is 2.7 points better at about the same size and with 22 times less computation. The paper's text rounds the gap to "4% better" (Section 4.2).

**Q6: Where does MobileNet spend its computation?**
A: 95% of the computation time goes into the 1×1 convolutions, which also hold 75% of the parameters (Section 3.2, Table 2). These can be computed directly as a matrix multiply (GEMM), one of the most optimised numerical routines, without the memory reordering that other convolutions need (Section 3.2).

## Common Misconceptions

1. **"Depthwise separable convolutions were invented by MobileNets."** The paper says they were introduced in Sifre's thesis and later used in Inception models, and it cites Flattened Networks, Factorized Networks and Xception as related work (Section 2). Its contribution is a whole family of networks built from them, with two simple knobs for size and speed.

2. **"Fewer multiply-adds always means faster."** Speed depends on how well the hardware and software run each operation. The paper points out that its design puts almost all the computation into 1×1 convolutions because they map to fast matrix multiplies (Section 3.2). In [this repo's notebook](solution.ipynb), the small MobileNet needs about 8× fewer multiply-adds but runs slower than the standard network on a T4 GPU at batch size 1.

3. **"Factorizing further always helps."** The paper argues that more factorization in the spatial dimensions, as in Flattened Networks or Inception V3, would not save much, because very little computation is left in the depthwise convolutions (Section 3.1). It does not report an experiment on this.

4. **"A toy reproduction shows ImageNet results."** The notebook trains a 5-layer network on 10,000 CIFAR-10 images for 4 epochs; its numbers only describe that small task. See its Summary.

## Real Citations

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)

2. Sifre, L. (2014). *Rigid-Motion Scattering for Image Classification.* PhD thesis.

3. Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017. [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)

4. Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.* [arXiv:1602.07360](https://arxiv.org/abs/1602.07360)

5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

6. Han, S., Mao, H., & Dally, W. J. (2016). *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)

**Citation count:** about 34,300 ([OpenAlex](https://openalex.org/W2612445135), 34,322 when checked on 27 November 2026).
