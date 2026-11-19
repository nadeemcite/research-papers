---
title: "DDPM: Talk & Q&A"
parent: "084 DDPM"
nav_order: 2
---
# TALK.md — Denoising Diffusion Probabilistic Models

## Press & Blog Coverage

1. [**"The Annotated Diffusion Model"**](https://huggingface.co/blog/annotated-diffusion), Hugging Face blog. Goes through the DDPM paper step by step and implements it in PyTorch.

2. [**"What are Diffusion Models?"**](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), Lilian Weng's blog. A technical overview of diffusion models that covers DDPM (Ho et al. 2020) alongside score-based models.

3. [**"Generative Modeling by Estimating Gradients of the Data Distribution"**](https://yang-song.net/blog/2021/score/), Yang Song's blog. Explains score-based generative models and says that in 2020 Jonathan Ho and colleagues significantly improved the empirical performance of diffusion probabilistic models.

4. [**"Introduction to Diffusion Models for Machine Learning"**](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction), AssemblyAI blog. A beginner-friendly explanation of diffusion models that lists the DDPM paper among its references.

## Q&A (answers from the paper)

**Q1: What does the network actually predict?**
A: The noise ε that was added to the image. The authors reparameterize the reverse-process mean so that the network ε_θ(x_t, t) predicts the noise, and the mean becomes (1/√α_t)(x_t − β_t/√(1 − ᾱ_t) · ε_θ(x_t, t)). They show this makes sampling resemble Langevin dynamics and makes the training objective resemble denoising score matching (Section 3.2).

**Q2: Is the reverse-process variance learned?**
A: No. The authors set it to untrained, time-dependent constants σ_t². Both σ_t² = β_t and σ_t² = β̃_t (the posterior variance) gave similar results (Section 3.2). In their ablation, learning a diagonal variance led to unstable training and poorer samples (Section 4.2, Table 2).

**Q3: What is the "simplified" objective?**
A: L_simple drops the weighting of the true variational bound and just takes the mean squared error between the real and predicted noise at a random timestep. Compared with the bound, it down-weights the easy denoising tasks at small t, which the authors found leads to better sample quality (Section 3.4). Training on the true bound gave better codelengths, but L_simple gave the best samples (Section 4.1).

**Q4: Which noise schedule did they use?**
A: T = 1000 steps, with β_t rising linearly from β_1 = 10⁻⁴ to β_T = 0.02, chosen to be small relative to data scaled to [−1, 1] (Section 4).

**Q5: How good were the samples?**
A: On unconditional CIFAR10 the model reached an Inception score of 9.46 and an FID of 3.17, better than most models in the literature, including class-conditional ones (Section 4.1, Table 1). In the ablation, predicting ε with L_simple was clearly the best combination (Section 4.2, Table 2).

**Q6: How large was the model?**
A: A U-Net based on PixelCNN++ with group normalization and self-attention at the 16×16 resolution; the CIFAR10 model has 35.7 million parameters (Section 4 and Appendix B).

## Common Misconceptions

1. **"Diffusion models were invented in this paper."** Diffusion probabilistic models come from Sohl-Dickstein et al. (2015), which this paper builds on (Section 2). Its contributions are the ε-prediction parameterization, the simplified objective and the high-quality results. The paper also shows a connection to denoising score matching (Section 3.2), the idea behind Song & Ermon's (2019) score-based models.

2. **"DDPM needs 1000 steps forever."** The paper samples with all T = 1000 steps. Later work, such as [DDIM](https://arxiv.org/abs/2010.02502), showed that faster sampling is possible.

3. **"The network learns both the mean and the variance."** The paper keeps the variance fixed; learning it was unstable in its ablation (Section 4.2, Table 2).

4. **"You need a huge model."** The paper's CIFAR10 model has 35.7M parameters (Section 4). In [this repo's notebook](solution.ipynb), a much smaller U-Net trained briefly on FashionMNIST shows the method at toy scale; see its Summary for the actual result.

## Real Citations

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

2. Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)

3. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)

4. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

5. Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)

6. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

7. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents.* [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

**Citation count:** about 20,800 ([OpenAlex](https://openalex.org/W3036167779), 20,776 when checked on 19 November 2026).
