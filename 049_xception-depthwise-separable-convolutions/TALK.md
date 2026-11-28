---
title: "Xception: Talk & Q&A"
parent: "049 Xception"
nav_order: 2
---
# TALK.md — Xception

## Press & Blog Coverage

1. [**Xception in Keras Applications**](https://keras.io/api/applications/xception/), Keras documentation. The open-source Keras implementation with ImageNet weights, which the paper points to as its reference implementation (Section 3).

2. [**tf.keras.applications.Xception**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception), TensorFlow documentation. The same model as part of TensorFlow's Keras API.

3. [**Xception on Hugging Face Papers**](https://huggingface.co/papers/1610.02357), the paper's page on Hugging Face (the old Papers With Code link now redirects here).

4. [**Xception at CVPR 2017**](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html), CVF Open Access. The conference version of the paper.

## Q&A (answers from the paper)

**Q1: What is the key insight behind Xception?**
A: An Inception module first maps cross-channel correlations with 1×1 convolutions into 3 or 4 smaller spaces, then maps spatial correlations inside each one (Section 1.1). A simplified Inception module is the same as a large 1×1 convolution followed by spatial convolutions on separate segments of its output channels (Section 1.2, Figures 2 and 3). With one segment per channel, this becomes almost exactly a depthwise separable convolution (Section 1.2, Figure 4). Xception, "Extreme Inception", is built on the stronger hypothesis that the two kinds of correlation can be mapped completely separately (Section 3).

**Q2: Why is there no ReLU between the depthwise and pointwise convolutions?**
A: The paper tested a ReLU or an ELU between the two steps on ImageNet and found that no non-linearity gave both faster convergence and better final performance (Section 4.7, Figure 10). This is the opposite of what Szegedy et al. reported for Inception modules. The paper suggests a non-linearity helps on deep feature spaces, as in Inception, but hurts on the 1-channel-deep spaces of depthwise convolutions, "possibly due to a loss of information" (Section 4.7).

**Q3: Xception has about as many parameters as Inception V3, so why is it better?**
A: The two have 22,855,952 and 23,626,728 parameters, within 3.5% of each other (Section 4.5.2, Table 3), so the paper says the gains come from using the parameters more efficiently rather than from more capacity. On ImageNet the gain is small: 79.0% vs 78.2% top-1 (Table 1). On JFT, with 350 million images and 17,000 classes, it is a 4.3% relative gain in MAP@100, 6.70 vs 6.36 (Section 4.5.1, Table 2). The paper suggests Inception V3, which was developed on ImageNet, "may thus be by design over-fit" to it (Section 4.5.1).

**Q4: How does Xception relate to MobileNets, which also use depthwise separable convolutions?**
A: The Xception paper mentions MobileNets as Andrew Howard's work within Google on efficient mobile models using depthwise separable convolutions, then forthcoming (Section 2), and the MobileNets paper says Xception "demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks" (its Section 2).

The designs differ. Xception has no non-linearity between the depthwise and pointwise steps (Section 4.7) and residual connections around most modules (Section 3). In the [MobileNets paper](https://arxiv.org/abs/1704.04861), both steps are followed by batchnorm and ReLU, it is a plain stack of layers, and it adds width and resolution multipliers to make the network smaller (its Sections 3.1 to 3.4, Table 1, Figure 3).

**Q5: What is the "discrete spectrum" the paper describes?**
A: Between a regular convolution and a depthwise separable convolution lies a spectrum, set by the number of independent channel segments used for the spatial convolutions. A regular convolution (preceded by a 1×1 convolution) is one segment, a depthwise separable convolution is one segment per channel, and Inception modules divide a few hundred channels into 3 or 4 segments (Section 1.2). The paper says there is no reason to believe the extreme end is optimal, and leaves the points in between for future work (Section 5).

**Q6: How much do the residual connections matter?**
A: On ImageNet, a version of Xception without residual connections converged more slowly and reached a worse result (Section 4.6, Figure 9). The paper adds that this only shows their importance for this architecture: non-residual VGG-style stacks of depthwise separable convolutions also gave excellent results, better than Inception V3 on JFT at equal parameter count (Section 4.6).

## Common Misconceptions

1. **"Xception invented depthwise separable convolutions."** The paper credits Laurent Sifre, who developed them during an internship at Google Brain in 2013, and notes their earlier use as the first layer of Inception V1 and V2 and in MobileNets (Section 2). Its contribution is the view of Inception modules as a point on a spectrum, and a full architecture built from these convolutions.

2. **"Xception is just a bigger MobileNet."** Both use depthwise separable convolutions, but Xception has residual connections and no non-linearity between the depthwise and pointwise steps (Sections 3 and 4.7). The MobileNets paper uses batchnorm and ReLU after both steps, a plain stack of layers and two multipliers for size (its Sections 3.1 to 3.4, Table 1, Figure 3).

3. **"The order of depthwise and pointwise steps matters."** The paper argues the order is unimportant, in particular because these operations are stacked (Section 1.2). It does not report an experiment on this.

4. **"Xception clearly beats Inception V3 on ImageNet."** The ImageNet gain is 0.8 points of top-1 accuracy (Table 1); the larger gain is on JFT (Table 2). The optimisation settings had been tuned for Inception V3, so the paper says they may be suboptimal for Xception, especially on ImageNet (Section 4.2).

5. **"A toy reproduction shows the paper's result."** [This repo's notebook](solution.ipynb) trains two small networks on CIFAR-10 for 3 epochs, and its Inception-style network is not the same size as its Xception. What it found is in its Summary.

## Real Citations

1. Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017. [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)

2. Sifre, L. (2014). *Rigid-Motion Scattering for Image Classification.* PhD thesis.

3. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision (Inception V3).* CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

5. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

6. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)

7. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

**Citation count:** about 24,400 ([OpenAlex](https://openalex.org/W2531409750), 24,405 when checked on 28 November 2026).
