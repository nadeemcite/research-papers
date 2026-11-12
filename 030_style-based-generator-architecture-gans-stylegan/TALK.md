---
title: "StyleGAN: Talk & Q&A"
parent: "030 StyleGAN"
nav_order: 2
---
# TALK — StyleGAN

## Press & Blog Coverage

1. [**"ThisPersonDoesNotExist.com uses AI to generate endless fake faces"**](https://www.theverge.com/tldr/2019/2/15/18226005/ai-generated-fake-people-portraits-thispersondoesnotexist-stylegan), The Verge, 15 February 2019. Coverage of the website that showed a new StyleGAN face on every page load.

2. [**NVlabs/stylegan**](https://github.com/NVlabs/stylegan), GitHub. The authors' official TensorFlow implementation.

3. [**NVlabs/ffhq-dataset**](https://github.com/NVlabs/ffhq-dataset), GitHub. The Flickr-Faces-HQ dataset released with the paper.

## Q&A (answers from the paper)

**Q1: What problem did the authors start from?**
A: The introduction says GAN generators "continue to operate as black boxes": the origin of features like stochastic detail is not understood, and the latent space is poorly understood. The authors redesign the generator, inspired by style transfer, to expose new ways to control image synthesis (Section 1).

**Q2: Why map z to an intermediate space W?**
A: The input latent space must follow the probability density of the training data, which the authors argue leads to some unavoidable entanglement. The intermediate space W is free from that restriction (Section 1). Figure 6 illustrates this with a training set missing one combination of features, such as long-haired males (Section 3.2).

**Q3: Why start from a learned constant?**
A: After adding the mapping network and AdaIN, the authors observed that the network no longer benefited from feeding the latent code into the first layer. So they start synthesis from a learned 4×4×512 constant (Section 2.1).

**Q4: What does the noise input do?**
A: Without it, a generator has to invent spatially varying pseudorandom numbers from earlier activations, which uses capacity and can leave repetitive patterns. Adding per-pixel noise after each convolution sidesteps this, and the noise affects only stochastic details such as the exact placement of hairs (Section 3.2).

**Q5: Why does each style affect only part of the image?**
A: AdaIN first normalizes each channel to zero mean and unit variance, and only then applies the style's scale and bias. So each style controls one convolution before the next AdaIN overrides it (Section 3).

## Common Misconceptions

1. **"StyleGAN fully disentangles its latent space."** The paper shows W is more linear and less entangled than a traditional generator's latent space, measured with its new perceptual path length and linear separability metrics (Sections 1 and 4). It does not claim full disentanglement.

2. **"StyleGAN invented AdaIN."** AdaIN comes from style-transfer work such as [Huang & Belongie (2017)](https://arxiv.org/abs/1703.06868), which the paper cites. StyleGAN applies it inside a GAN generator, with styles computed from W.

3. **"The face results come from a quick training run."** Appendix C reports about one week of training on an NVIDIA DGX-1 with 8 Tesla V100 GPUs for the high-resolution results.

4. **"Noise and styles do the same job."** In the paper, styles change high-level attributes, while noise only changes stochastic details and leaves the overall composition intact (Section 3.2).

## Real Citations

- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR 2019. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
- Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). *Progressive Growing of GANs for Improved Quality, Stability, and Variation.* ICLR 2018. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
- Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
- Karras, T., et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN.* [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
- Karras, T., et al. (2021). *Alias-Free Generative Adversarial Networks.* [arXiv:2106.12423](https://arxiv.org/abs/2106.12423)

**Citation count:** about 12,600 ([OpenAlex](https://openalex.org/W2904367110), 12,598 when checked on 12 November 2026).
