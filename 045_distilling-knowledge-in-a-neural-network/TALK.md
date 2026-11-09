---
title: "Distillation: Talk & Q&A"
parent: "045 Distillation"
nav_order: 2
---
# TALK.md — Distilling the Knowledge in a Neural Network

## Press & Blog Coverage

The links below are papers that build on or survey this work. No press coverage was verified for this page, so none is listed.

1. [**Knowledge Distillation: A Survey**](https://arxiv.org/abs/2006.05525), Gou et al. A broad survey covering this method and its extensions.

2. [**DistilBERT, a distilled version of BERT**](https://arxiv.org/abs/1910.01108), Sanh et al. A well-known application of distillation to a Transformer.

## Q&A (answers from the paper)

**Q1: What is the core idea?**
A: A cumbersome model's probabilities carry information about which wrong answers are more likely than others. Distillation trains a small model to match those soft targets, so it learns to generalize like the cumbersome model (Sections 1 and 2).

**Q2: Why raise the temperature?**
A: A higher temperature in the softmax produces a softer distribution over classes. The same high temperature is used while training the distilled model, and afterwards it uses a temperature of 1 (Section 2).

**Q3: Why multiply the soft loss by T squared?**
A: The gradients from the soft targets scale as 1/T². Multiplying by T² keeps the relative contributions of the hard and soft targets roughly unchanged when the temperature changes (Section 2).

**Q4: What did the MNIST experiment show?**
A: A large net with two 1200-unit hidden layers made 67 test errors. A smaller net with two 800-unit layers and no regularization made 146. Trained on soft targets from the large net at a temperature of 20, the small net made 74 (Section 3).

**Q5: How does this relate to matching logits?**
A: In the high-temperature limit, distillation is equivalent to minimizing ½(zᵢ − vᵢ)², provided the logits are zero-meaned separately for each transfer case. At lower temperatures, it pays much less attention to logits far below the average (Section 2.1).

## Common Misconceptions

1. **"Higher temperature is always better."** With 300 or more units per layer, all temperatures above 8 gave fairly similar results. With 30 units per layer, temperatures from 2.5 to 4 worked significantly better than higher or lower ones (Section 3).

2. **"You need an ensemble as the teacher."** The paper's speech experiment distils an ensemble of 10 models (Section 4), but its MNIST experiment uses a single large net (Section 3).

3. **"A distilled model always beats one trained on hard labels."** The paper reports clear gains, but small setups may not. In [this repo's notebook](solution.ipynb) the result on a small MLP is much narrower, as its Summary cell reports.

## Real Citations

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NIPS 2014 Deep Learning Workshop, now NeurIPS. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Ba, L. J., & Caruana, R. (2014). Do Deep Nets Really Need to Be Deep? NIPS 2014, now NeurIPS. [arXiv:1312.6184](https://arxiv.org/abs/1312.6184)
- Romero, A., et al. (2015). FitNets: Hints for Thin Deep Nets. ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
- Gou, J., et al. (2021). Knowledge Distillation: A Survey. [arXiv:2006.05525](https://arxiv.org/abs/2006.05525)
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

**Citation count:** about 25,400 ([OpenAlex](https://openalex.org/W1821462560), checked November 2026).
