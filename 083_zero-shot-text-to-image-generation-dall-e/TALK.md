---
title: "DALL-E: Talk & Q&A"
parent: "083 DALL-E"
nav_order: 2
---
# TALK.md — DALL-E (Zero-Shot Text-to-Image Generation)

## Press & Blog Coverage

1. [**DALL·E: Creating images from text**](https://openai.com/index/dall-e/), OpenAI, 5 January 2021. OpenAI's announcement of DALL-E, with interactive examples such as "an armchair in the shape of an avocado".

2. [**OpenAI debuts DALL-E for generating images from text**](https://venturebeat.com/technology/openai-debuts-dall-e-for-generating-images-from-text), Khari Johnson, VentureBeat, 5 January 2021. Reports the launch of DALL-E together with CLIP, a network trained on 400 million image-text pairs.

3. [**OpenAI's DALL-E app generates images from just a description**](https://www.engadget.com/dall-e-ai-gpt-make-image-from-any-description-135535140.html), Steve Dent, Engadget, 6 January 2021. Describes examples such as "a cat made of sushi" and "a pentagonal green clock", and its use for image-to-image translation guided by a caption, with a cat and a sketch of it.

4. [**CLIP: Connecting text and images**](https://openai.com/index/clip/), OpenAI, January 2021. The announcement of CLIP, the contrastive model that DALL-E uses to rerank its samples.

## Q&A (answers from the paper)

**Q1: Why does DALL-E need a discrete VAE?**
A: Using pixels directly as tokens would need too much memory for high-resolution images, and likelihood training would spend most of the model on fine pixel detail. The dVAE compresses each 256×256 image into a 32×32 grid of tokens, each with 8192 possible values, which shrinks the transformer's context by a factor of 192 "without a large degradation in visual quality" (Section 2).

**Q2: How is the dVAE trained, if its codes are discrete?**
A: A discrete choice has no gradient. Instead of the straight-through estimator used in VQ-VAE, DALL-E uses the gumbel-softmax relaxation, which becomes exact as its temperature τ goes to 0; annealing τ to 1/16 was enough (Section 2.1). Raising the KL weight to β = 6.6 gave better codebook use and, unusually, a smaller final reconstruction error (Section 2.1).

**Q3: What exactly does the transformer model?**
A: Up to 256 BPE tokens of the lowercased caption (vocabulary 16,384) followed by the 1024 image tokens (vocabulary 8192), as one stream. It is a 12-billion-parameter sparse transformer with 64 self-attention layers. The image tokens come from the argmax of the dVAE encoder's logits, and the text and image losses are weighted 1/8 and 7/8 (Section 2.2).

**Q4: What data was it trained on?**
A: 250 million text-image pairs collected from the internet, including Conceptual Captions and a filtered subset of YFCC100M, but not MS-COCO. About 21% of the MS-COCO validation images turned out to be in the training data (none of the captions), and removing them made no significant difference to the results (Sections 2.3 and 3.1).

**Q5: How good were the results?**
A: On MS-COCO captions, in a best-of-five vote, human judges preferred DALL-E's sample to DF-GAN's 93% of the time for matching the caption and 90% for realism. Its FID was within 2 points of the best prior method, and it did much worse on CUB birds, with a gap of nearly 40 FID points (Section 3.1).

**Q6: What role does reranking play?**
A: Several images are drawn for each caption and a pretrained contrastive model scores how well each matches the caption. Unless stated otherwise the paper's samples are the best of 512 (Section 2.6), and FID and IS on MS-COCO improve as the number of samples grows, up to about 32 (Section 3.1).

## Common Misconceptions

1. **"DALL-E is a diffusion model."** This paper's model is an autoregressive transformer over dVAE tokens (Section 2). The later [DALL·E 2](https://arxiv.org/abs/2204.06125) generates images with a diffusion decoder.

2. **"The dVAE is a VQ-VAE with a commitment loss."** It is trained with the gumbel-softmax relaxation and a KL term to a uniform prior, which the paper contrasts with the straight-through estimator of VQ-VAE (Section 2.1).

3. **"Zero-shot means DALL-E never saw anything like the test images."** In the paper, zero-shot means it was evaluated on datasets such as MS-COCO without training on them (Section 3.1). Its training data did include about 21% of the MS-COCO validation images, without their captions (Section 3.1).

4. **"DALL-E composes concepts reliably."** The paper says it combines concepts and binds attributes "inconsistently", for example sometimes putting the Christmas sweater on both animals (Section 3.3).

5. **"A toy reproduction shows DALL-E's abilities."** [This repo's notebook](solution.ipynb) trains a tiny model on synthetic coloured shapes; what it can and cannot do is described in its Summary.

## Real Citations

1. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., & Sutskever, I. (2021). *Zero-Shot Text-to-Image Generation.* ICML 2021. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)

2. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning (VQ-VAE).* NIPS 2017 (now NeurIPS). [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

3. Jang, E., Gu, S., & Poole, B. (2017). *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)

4. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). *Generating Long Sequences with Sparse Transformers.* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)

5. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

6. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL·E 2).* [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

7. Yu, J., Xu, Y., Koh, J. Y., et al. (2022). *Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti).* [arXiv:2206.10789](https://arxiv.org/abs/2206.10789)

**Citation count:** about 6,700 ([OpenAlex](https://openalex.org/W3129576130), 6,685 when checked on 26 November 2026).
