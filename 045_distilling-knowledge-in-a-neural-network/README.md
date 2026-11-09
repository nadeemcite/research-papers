---
title: "045 Distillation"
nav_order: 46
permalink: "/045_distilling-knowledge-in-a-neural-network/"
---
# Distilling the Knowledge in a Neural Network
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/distilling-the-knowledge-in-a-neural-network)

- **Paper:** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- **Lab, year:** Google, 2015 (NIPS 2014 Deep Learning Workshop, now NeurIPS)

## Summary

Knowledge distillation trains a small model to copy what a large, cumbersome model has learned. A trained model's output probabilities (its softmax values, the percentages it gives each class) say more than which answer is right. They also show which wrong answers it considers more likely.

Those relative probabilities are often tiny, so the paper raises a "temperature" in the softmax to soften them. The small model trains on these soft targets, usually alongside the true labels.

On MNIST, a small net made 146 test errors on hard labels. Distilled from a larger net that made 67 errors, it made 74. On a speech recognition model close to the one used by Android voice search, more than 80% of the improvement from an ensemble of 10 models carried over to a single distilled model.

## What Problem Does It Solve

Imagine a master chef who has spent decades learning to cook. The chef is brilliant but slow, and needs a huge kitchen. You want a fast-food chain whose cooks serve nearly the same food in small kitchens.

If you only tell the new cooks what the finished dish should be called, they learn slowly. But if the master says "this is 90% chicken broth, 8% vegetable, 2% something else", the cooks also learn what each dish is similar to.

Large models that do well are expensive to run for many users. Distillation passes on their judgment to a model small enough to deploy.

## Core Idea

- **Soft targets:** train the small model on the large model's class probabilities, not only on the correct label.
- **Temperature:** the softmax becomes q_i = exp(z_i / T) / sum_j exp(z_j / T). A higher T gives a softer distribution. The distilled model trains at that high T and then uses T = 1.
- **Two objectives:** a weighted average of cross-entropy with the soft targets and cross-entropy with the true labels. The paper generally got the best results with a considerably lower weight on the second one.

## Key Method Details

- **T² scaling:** the soft-target gradients scale as 1/T², so the soft loss is multiplied by T² to keep the two objectives' relative contributions roughly unchanged.
- **Matching logits is a special case:** in the high-temperature limit, with zero-meaned logits, distillation is equivalent to minimizing the squared difference between the two models' logits (their scores before the softmax).
- **Choosing the temperature:** with 300 or more units per layer, all temperatures above 8 gave fairly similar results on MNIST. With 30 units per layer, 2.5 to 4 worked significantly better.

### Simplifications in This Notebook

The notebook uses a single 800-800-10 teacher with dropout and a 300-300-10 student, on MNIST, on the CPU. The paper's MNIST large net has 1200 units per layer and its small net 800. The main runs use T = 4 and alpha = 0.7 for 10 epochs, with a sweep over T = 1, 2, 4, 8, 16 and 20.

### What the Notebook Result Shows

Final test accuracy: teacher **98.00%**, student on hard labels **97.85%**, distillation only **97.78%**, combined loss **97.92%**.

Distillation did not clearly help in this run. Distillation alone ended 0.07 points below the hard-label student, and the combined loss ended 0.07 points above it. A second student with the same settings but different starting weights reached 98.01%, so these gaps are within the noise. The teacher is only 0.15 points ahead of the student, too small a gap to show the paper's effect.

The temperature sweep ranged from 97.67% to 98.01% with no clear trend. On one batch from an untrained student, soft targets had a *higher* gradient variation (0.192) than hard targets (0.155), which does not reproduce the paper's statement.

## Influence

OpenAlex lists about 25,400 citations (checked November 2026). The paper's teacher-student recipe is widely built on, including [FitNets](https://arxiv.org/abs/1412.6550) and [DistilBERT](https://arxiv.org/abs/1910.01108), and [a survey by Gou et al.](https://arxiv.org/abs/2006.05525) covers the field that grew around it.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/045_distilling-knowledge-in-a-neural-network/solution.ipynb) | From-scratch PyTorch implementation of knowledge distillation on MNIST, with a temperature sweep and plots |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Citations, Q&A (answers from the paper), and common misconceptions about the paper |
