---
title: "pix2pix: Talk & Q&A"
parent: "025 pix2pix"
nav_order: 2
---
# TALK.md — Pix2Pix Press, Q&A, and Citations

## Press / Blog Coverage

- **Two Minute Papers:** [video by Károly Zsolnai-Fehér](https://www.youtube.com/watch?v=u7kQ5lNfUfg), linked from the [official project page](https://phillipi.github.io/pix2pix/).
- **Affine Layer (Christopher Hesse):** [Image-to-Image Translation in Tensorflow](https://affinelayer.com/pix2pix/) — a detailed post on his TensorFlow port, which also powered the #edges2cats interactive demo.
- **The Verge:** [Please meet my terrible fur sons](https://www.theverge.com/2017/2/21/14678230/edge2cat-machine-learning-tensorflow-ai-drawing-cthulu-take-the-wheel) — coverage of the #edges2cats demo.
- **Vox:** [This bizarre image-translation tool turns drawings into photographs](https://www.vox.com/2017/2/23/14702058/cat-image-machine-learning) — coverage of the same demo.
- **Gizmodo:** [Turn Doodles Into Furry Cat Monsters With Machine Learning Algorithm](https://gizmodo.com/turn-doodles-into-furry-cat-monsters-with-machine-learn-1792629969).

## Q&A (answers from the paper)

**Q: What was the key insight that made Pix2Pix work as a general-purpose solution?**
A (from the paper, Section 1): Instead of hand-crafting a different loss function for each image translation task, the authors let a conditional GAN learn the loss from the data. The discriminator learns what looks real for the task at hand, so the same setup works across different tasks: only the paired training data changes.

**Q: Why use a U-Net generator rather than a standard encoder-decoder?**
A (from the paper, Sections 3.2.1 and 4.3, Table 2): In image-to-image translation, the input and output share a lot of low-level structure, such as the location of edges. An encoder-decoder forces all information through a bottleneck, while skip connections shuttle it directly across. On Cityscapes labels→photos with L1+cGAN, the U-Net scores 0.55 per-pixel accuracy against 0.29 for the encoder-decoder.

**Q: What is a PatchGAN, and why use it rather than a full-image discriminator?**
A (from the paper, Sections 3.2.2 and 4.4, Table 3): A PatchGAN classifies N×N patches rather than the whole image, modelling the image as a Markov random field that assumes independence beyond a patch diameter. It can be understood as a texture/style loss, has fewer parameters, runs faster, and works on arbitrarily large images. In Table 3, the 70×70 discriminator gets the best per-pixel accuracy (0.66), ahead of 16×16 (0.65), 286×286 (0.42) and 1×1 (0.39).

**Q: Why combine L1 loss with the GAN loss?**
A (from the paper, Sections 3.2.2 and 4.2, Table 1): L1 alone gives blurry results, and cGAN alone gives sharper results with some visual artifacts. L1 captures the low frequencies while the GAN enforces high-frequency realism. On Cityscapes, the per-pixel accuracy is 0.42 for L1, 0.57 for cGAN and 0.66 for L1+cGAN.

**Q: How much data does it need?**
A (from the paper, Section 4): Often not much. The facade training set has just 400 images and the day-to-night set only 91 unique webcams. The facade results took less than two hours of training on a single Pascal Titan X GPU.

## Common Misconceptions

1. **"Pix2Pix needs unpaired data"** — No. Pix2Pix requires *paired* data (input + target). It's CycleGAN (the follow-up by the same group) that works with unpaired data.

2. **"The noise vector z is important"** — In practice, the generator learned to ignore explicit noise input. The authors found dropout more effective for providing stochasticity, and even then the outputs showed only minor stochastic variation.

3. **"PatchGAN sees the whole image"** — No. Each patch is classified independently. The 70×70 PatchGAN only looks at 70×70 pixel regions. This is why it acts as a local texture loss rather than a global structure loss.

4. **"You need a huge dataset"** — The paper explicitly shows good results with as few as 91-400 training images for some tasks.

## Real Citations

- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004).
- Follow-up work: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (CycleGAN, ICCV 2017), Wang et al., "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs" (pix2pixHD, CVPR 2018).
- Official code: [phillipi/pix2pix](https://github.com/phillipi/pix2pix) (Torch) and [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (PyTorch).
- Project page: [phillipi.github.io/pix2pix](https://phillipi.github.io/pix2pix/).
