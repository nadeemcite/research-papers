---
title: "ProGAN: Talk & Q&A"
parent: "029 ProGAN"
nav_order: 2
---
# TALK — Progressive Growing of GANs

## Press & Blog Coverage

1. [**Progressive Growing of GANs for Improved Quality, Stability, and Variation**](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of), NVIDIA Research. The lab's publication page for the paper.

2. [**tkarras/progressive_growing_of_gans**](https://github.com/tkarras/progressive_growing_of_gans), GitHub. The authors' official TensorFlow implementation. Its README says the original Theano version produced the paper's results.

3. [**"How to Implement Progressive Growing GAN Models in Keras"**](https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/), Machine Learning Mastery. A step-by-step tutorial covering fade-in and the toRGB/fromRGB layers.

## Q&A (answers from the paper)

**Q1: What is the main idea?**
A: Start with low-resolution images and add layers to both networks as training goes on. The authors say this lets training first discover the large-scale structure of the image distribution, and then shift attention to finer detail, instead of learning all scales at once (Section 2).

**Q2: Why fade new layers in?**
A: New layers are blended in smoothly while the resolution doubles. The authors say this avoids sudden shocks to the already well-trained, smaller-resolution layers (Section 2 and Figure 2).

**Q3: What loss did the authors train with?**
A: The WGAN-GP loss, with the generator and discriminator updated alternately on each minibatch. They also add a small term that keeps the discriminator output from drifting too far from zero, with weight 0.001 (Appendix A.1). Section 2 adds that the method also trains with LSGAN loss, though Appendix B says they prefer WGAN-GP.

**Q4: How does minibatch standard deviation work?**
A: The layer computes the standard deviation of each feature at each spatial location over the minibatch, averages them into one value, and appends it as an extra constant feature map. It has no learnable parameters, and the authors insert it near the end of the discriminator (Section 3).

**Q5: Why scale weights at runtime instead of initializing them carefully?**
A: Weights start from N(0, 1) and are scaled by the per-layer constant from He's initializer at every forward pass. Optimizers like Adam normalize updates by their scale, so this keeps the learning speed the same for all weights (Section 4.1).

## Common Misconceptions

1. **"ProGAN uses transposed convolutions to upsample."** The paper doubles the resolution with nearest-neighbor filtering, and Appendix A.1 describes upsampling as 2×2 element replication, each followed by convolutions. This notebook does the same.

2. **"ProGAN uses the standard non-saturating GAN loss."** The paper trains with WGAN-GP loss (Appendix A.1). This notebook uses the non-saturating loss as a simplification.

3. **"The 1024×1024 faces come from CelebA."** They come from CelebA-HQ, a higher-quality version of CelebA that the authors built for the paper (Section 6.3).

4. **"Later NVIDIA GANs all kept progressive growing."** [StyleGAN2](https://arxiv.org/abs/1912.04958) revisited it and reports a design that produces high-quality images without progressive growing (Section 4 of that paper).

## Real Citations

- Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). *Progressive Growing of GANs for Improved Quality, Stability, and Variation.* ICLR 2018. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
- Gulrajani, I., et al. (2017). *Improved Training of Wasserstein GANs.* [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) (the WGAN-GP loss used in the paper)
- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks.* [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
- Karras, T., et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN.* [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
- Brock, A., Donahue, J., & Simonyan, K. (2019). *Large Scale GAN Training for High Fidelity Natural Image Synthesis.* [arXiv:1809.11096](https://arxiv.org/abs/1809.11096)

**Citation count:** about 7,000 ([OpenAlex](https://openalex.org/W2766527293), 7,024 when checked on 10 November 2026).
