---
title: "CycleGAN: Talk & Q&A"
parent: "026 CycleGAN"
nav_order: 2
---
# Talk / Press / Coverage — CycleGAN

## Press and blog coverage

- **Project page:** [CycleGAN Project Page](https://junyanz.github.io/CycleGAN/) — the authors' page, with results for horse to zebra, summer to winter and photo to painting, plus links to the code.
- **Authors' code:** [junyanz/CycleGAN on GitHub](https://github.com/junyanz/CycleGAN) — the original implementation released with the paper.
- **IEEE Signal Processing Society newsletter:** [AI learns to synthesize zebras from horses and vice versa](https://signalprocessingsociety.org/our-story/news-updates/2017/10/ai-learns-synthesize-zebras-horses-and-vice-versa) — an October 2017 news item on the Berkeley work.
- **TensorFlow tutorial:** [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan) — a step-by-step implementation that trains on the horse and zebra images.

## Q&A (answers from the paper)

**Q: What is the hard part of learning without paired data?**

A (from the paper, Section 1): The adversarial objective only says the outputs should look like the target domain as a whole. Many different mappings can produce that same output distribution, and nothing ties a particular input to a particular output. The optimizer can also collapse so that every input maps to the same output. Requiring the two mappings to undo each other narrows down the possibilities.

**Q: Why use a PatchGAN discriminator instead of judging the whole image?**

A (from the paper, Section 4, and the Pix2Pix paper, Section 3.2.2): CycleGAN uses a 70×70 PatchGAN because a patch-level discriminator has fewer parameters than a full-image one and can work on images of any size. The reasoning comes from Pix2Pix: L1 loss already captures low-frequency correctness, so the GAN only needs to model high-frequency structure, and local patches are enough for that.

**Q: When does CycleGAN fail?**

A (from the paper, Section 6): It often succeeds on color and texture changes, but the authors explored geometric changes with little success. On dog to cat, the learned translation degenerates into minimal changes to the input. They suggest this may come from generator architectures tailored to appearance changes. Other failures come from the training data: a horse-to-zebra model trained on ImageNet "wild horse" images fails on a person riding a horse.

**Q: What does the identity loss do?**

A (from the paper, Section 5.2): On painting to photo, the generators tended to change the tint of images when there was no need. Following Taigman et al., the authors add a term that keeps a generator close to an identity mapping when it is given real images of its target domain, which helps preserve the color of the input. Its weight is 0.5λ, and the paper describes using it for specific tasks such as this one, not all of them.

**Q: How does CycleGAN compare to Pix2Pix?**

A (from the paper, Section 5.1): The authors compare against Pix2Pix as an "upper bound", since Pix2Pix trains on paired data. On Cityscapes labels to photo, Pix2Pix still scores higher (FCN per-pixel accuracy of 0.71 against CycleGAN's 0.52), so giving up the pairs does cost accuracy.

## Common misconceptions

- **"CycleGAN can translate any domain to any domain."** Section 6 of the paper reports little success on geometric changes such as dog to cat.
- **"Cycle consistency guarantees perfect reconstruction."** It is a weighted L1 penalty, not a hard constraint, so reconstructions are close but not exact.
- **"CycleGAN was the first method of its kind."** DualGAN and DiscoGAN were concurrent work that used similar two-way reconstruction ideas.
- **"The identity loss is part of every CycleGAN."** The paper uses it only for certain tasks, and this notebook turns it on throughout as a demonstration.

## Citations

- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593). ICCV 2017.
- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). CVPR 2017.
- Yi, Z., Zhang, H., Tan, P., & Gong, M. (2017). [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510). ICCV 2017. Concurrent work.
- Kim, T., Cha, M., Kim, H., Lee, J. K., & Kim, J. (2017). [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192). ICML 2017. DiscoGAN, concurrent work.
- Choi, Y., et al. (2018). [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020). CVPR 2018.
- [OpenAlex record for the paper](https://openalex.org/W2962793481) — 24,419 citations when checked on 9 Nov 2026.
