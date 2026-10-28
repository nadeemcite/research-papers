---
title: "Highway Nets: Talk & Q&A"
parent: "009 Highway Nets"
nav_order: 2
---
# Highway Networks — Talk / Context

## Verifiable coverage

- **Original arXiv preprint:** https://arxiv.org/abs/1505.00387 (ICML 2015 Deep Learning workshop extended abstract).
- **Full follow-up paper:** *Training Very Deep Networks* — https://arxiv.org/abs/1507.06228 (authors: Srivastava, Greff, Schmidhuber).
- **Author institutional affiliation:** The Swiss AI Lab IDSIA / Università della Svizzera italiana (USI) / SUPSI, as listed on the paper's title page: https://arxiv.org/abs/1505.00387
- **Acknowledged funding/hardware:** EU project NASCENCE (FP7-ICT-317662) and NVIDIA Tesla K40 GPU donation, per the Acknowledgments of https://arxiv.org/abs/1505.00387
- **Related work the paper builds on or compares with:** [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) initialization, [He et al. (2015)](https://arxiv.org/abs/1502.01852) initialization, [Simonyan & Zisserman (2014)](https://arxiv.org/abs/1409.1556) very deep convolutional networks, and [FitNets (Romero et al., 2014)](https://arxiv.org/abs/1412.6550). The paper does not cite batch normalization or ResNet; ResNet came later.

## Q&A (answers from the paper)

**Q1: What makes Highway Networks different from a normal deep MLP?**  
From the paper (Section 2): a normal layer always transforms its input through weights and a non-linearity. A Highway layer adds a *transform gate* `T(x)` and a *carry gate* `1 − T(x)`. It can choose to keep its input unchanged, transform it, or blend the two. This gives the network “information highways” along which data can travel many layers without being distorted.

**Q2: Why can Highway Networks be trained with so many layers?**  
Two reasons. First, the gates are initialized so that the network starts in a near-identity mode (bias of the transform gate is set to a negative value such as −2 or −4), so the early training signal can flow through. Second, during training the gates learn to open only where transformation is useful, preserving useful signals across depth. (Section 2 and Section 3.1 of the paper.)

**Q3: Are Highway Networks still used today?**  
They are rarely used as-is in modern production pipelines, but they are historically important as a direct conceptual precursor to ResNet. ResNet’s residual block `y = F(x) + x` is essentially a Highway block in which the carry path is always fully open and only a learned residual is added. This is later context, not from the paper; see the ResNet paper: https://arxiv.org/abs/1512.03385

**Q4: What kind of activation can the non-linear block use?**  
The paper experiments with ReLU and tanh, and notes that the negative transform-gate bias initialization works across different activation functions and different zero-mean initializations of `W_H` (Section 2 and Section 3.1).

**Q5: Did the authors release code?**  
Yes, with the full paper. The short workshop paper does not mention code, but the full version, *Training Very Deep Networks* (Section 3, https://arxiv.org/abs/1507.06228), says the experiments used Caffe and Brainstorm (https://github.com/IDSIA/brainstorm) and that source code, hyperparameter search results and scripts were published on the first author's IDSIA web page.

## Common misconceptions

- **Misconception:** “Highway Networks and ResNet are unrelated.”  
  **Correction:** ResNet's `y = F(x) + x` is like a highway block whose carry path is always open, with no gate.
- **Misconception:** “Highway layers increase parameters massively.”  
  **Correction:** Each Highway layer has two weight matrices (`W_H` and `W_T`), but the paper matches parameter counts to plain networks by adjusting layer widths for fair comparison.
- **Misconception:** “The transform gate must always be 0 or 1.”  
  **Correction:** The sigmoid output is in `(0, 1)`; it is a smooth interpolation, not a hard switch.

## Real citations

1. R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway Networks,” arXiv:1505.00387 [cs.LG], 2015. Presented at ICML 2015 Deep Learning Workshop.
2. R. K. Srivastava, K. Greff, and J. Schmidhuber, “Training Very Deep Networks,” arXiv:1507.06228 [cs.LG], 2015.
3. K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” arXiv:1502.01852, 2015.
4. A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “FitNets: Hints for Thin Deep Nets,” arXiv:1412.6550, 2014.
5. X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” AISTATS, 2010.
6. K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv:1409.1556, 2014.
