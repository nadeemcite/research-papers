---
title: "VAE: Talk & Q&A"
parent: "021 VAE"
nav_order: 2
---
# TALK.md — Auto-Encoding Variational Bayes (VAE)

## Press & Blog Coverage

1. [**Tutorial on Variational Autoencoders**](https://arxiv.org/abs/1606.05908), Carl Doersch (2016). A widely read introduction to the math and intuition behind VAEs.

2. [**Deep Generative Modeling**](https://link.springer.com/book/10.1007/978-3-030-93158-2), Jakub Tomczak (Springer, 2022). A textbook on deep generative models, including VAEs and their extensions.

3. [**"From Autoencoder to Beta-VAE"**](https://lilianweng.github.io/posts/2018-08-12-vae/), Lilian Weng (2018). A blog post tracing the path from plain autoencoders to VAEs and β-VAE, including the reparameterization trick.

4. [**An Introduction to Variational Autoencoders**](https://arxiv.org/abs/1906.02691), Kingma & Welling (2019). The authors' own survey of VAEs and the follow-up work.

## Q&A (answers from the paper)

**Q1: What problem does the paper address?**
A: How to do efficient inference and learning in directed probabilistic models whose latent variables have intractable posterior distributions, on large datasets (Abstract and Section 1).

**Q2: What is the reparameterization trick?**
A: A random variable z drawn from q(z\|x) is rewritten as a deterministic function of x and an independent noise variable ε, z = g(ε, x). The Monte Carlo estimate of the lower bound then becomes differentiable with respect to the variational parameters (Section 2.4).

**Q3: Does it only work for Gaussians?**
A: No. Section 2.4 lists three ways to choose g: a tractable inverse CDF, a "location-scale" family such as the Gaussian, and composition of simpler variables.

**Q4: Why "auto-encoder"?**
A: In the objective, the KL term acts as a regularizer and the second term is an expected negative reconstruction error. The recognition model q(z\|x) plays the encoder and p(x\|z) the decoder (Sections 2.3 and 3).

**Q5: How was it tested?**
A: On MNIST and Frey Face, with encoders and decoders of 500 and 200 hidden units, minibatches of 100 and one sample per datapoint. AEVB optimized the lower bound faster than the wake-sleep algorithm and reached a better solution in all experiments (Section 5).

## Common Misconceptions

1. **"A VAE is just an autoencoder."** A plain autoencoder only minimizes reconstruction error. The VAE optimizes a variational lower bound on the data likelihood, with a prior over latents that lets it generate new samples (Sections 2.2 and 3).

2. **"The KL term is an arbitrary regularizer."** It is part of the variational lower bound, and for a Gaussian prior and posterior it can be computed in closed form (Section 2.3 and Appendix B).

3. **"The trick only works for Gaussian latents."** The paper gives three general strategies (Section 2.4).

4. **"VAEs always give blurry images."** Later hierarchical VAEs such as [NVAE](https://arxiv.org/abs/2007.03898) and [Very Deep VAEs](https://arxiv.org/abs/2011.10650) report much stronger image results.

5. **"DALL-E used a VQ-VAE."** The original DALL-E trained a discrete VAE (dVAE), from the same discrete-latent family as VQ-VAE ([DALL-E paper](https://arxiv.org/abs/2102.12092)).

## Real Citations

1. Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models.* ICML 2014. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)

3. Higgins, I., et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017. [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)

4. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning* (VQ-VAE). NIPS 2017, now NeurIPS. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

5. Vahdat, A., & Kautz, J. (2020). *NVAE: A Deep Hierarchical Variational Autoencoder.* NeurIPS 2020. [arXiv:2007.03898](https://arxiv.org/abs/2007.03898)

6. Child, R. (2020). *Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images* (VDVAE). [arXiv:2011.10650](https://arxiv.org/abs/2011.10650)

7. Ramesh, A., et al. (2021). *Zero-Shot Text-to-Image Generation* (DALL-E). [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)

8. Doersch, C. (2016). *Tutorial on Variational Autoencoders.* [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)

**Citation count:** about 38,250 ([OpenAlex](https://openalex.org/W1959608418), 38,253 when checked on 15 November 2026).
