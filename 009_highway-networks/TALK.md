# Highway Networks — Talk / Context

## Verifiable coverage

- **Original arXiv preprint:** https://arxiv.org/abs/1505.00387 (ICML 2015 Deep Learning workshop extended abstract).
- **Full follow-up paper:** *Training Very Deep Networks* — https://arxiv.org/abs/1507.06228 (authors: Srivastava, Greff, Schmidhuber).
- **Author institutional affiliation:** The Swiss AI Lab IDSIA / Università della Svizzera italiana (USI) / SUPSI.
- **Acknowledged funding/hardware:** EU project NASCENCE (FP7-ICT-317662) and NVIDIA Tesla K40 GPU donation.
- **Related contemporaneous work cited in the paper:** FitNets (Romero et al., 2014), He initialization (He et al., 2015), batch normalization (Ioffe & Szegedy, 2015), ResNet (He et al., 2015) — ResNet was published shortly afterward and can be seen as a special case where the carry gate is always open.

## Interview-style Q&A

**Q1: What makes Highway Networks different from a normal deep MLP?**  
A normal layer always transforms its input through weights and a non-linearity. A Highway layer adds a *transform gate* `T(x)` and a *carry gate* `1 − T(x)`. It can choose to keep its input unchanged, transform it, or blend the two. This gives the network “information highways” along which data can travel many layers without being distorted.

**Q2: Why can Highway Networks be trained with so many layers?**  
Two reasons. First, the gates are initialized so that the network starts in a near-identity mode (bias of the transform gate is set to a negative value such as −2 or −4), so the early training signal can flow through. Second, during training the gates learn to open only where transformation is useful, preserving useful signals across depth.

**Q3: Are Highway Networks still used today?**  
They are rarely used as-is in modern production pipelines, but they are historically important as a direct conceptual precursor to ResNet. ResNet’s residual block `y = F(x) + x` is essentially a Highway block in which the carry path is always fully open and only a learned residual is added.

**Q4: What kind of activation can the non-linear block use?**  
The paper experiments with ReLU and tanh, and notes that the negative transform-gate bias initialization works across different activation functions and different zero-mean initializations of `W_H`.

**Q5: Did the authors release code?**  
The ICML workshop abstract does not mention public code; the full paper (arXiv:1507.06228) extends the experiments with additional references and analysis. Implementations have been reproduced by the community in PyTorch and TensorFlow.

## Common misconceptions

- **Misconception:** “Highway Networks and ResNet are unrelated.”  
  **Correction:** ResNet is closely related; a Highway block with `T(x) = 1/2` everywhere is not ResNet, but a block with a fixed fully-open carry path is mathematically equivalent to a residual block with `F(x) = H(x)/2`.
- **Misconception:** “Highway layers increase parameters massively.”  
  **Correction:** Each Highway layer has two weight matrices (`W_H` and `W_T`), but the paper matches parameter counts to plain networks by adjusting layer widths for fair comparison.
- **Misconception:** “The transform gate must always be 0 or 1.”  
  **Correction:** The sigmoid output is in `(0, 1)`; it is a smooth interpolation, not a hard switch.

## Real citations

1. R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway Networks,” arXiv:1505.00387 [cs.LG], 2015. Presented at ICML 2015 Deep Learning Workshop.
2. R. K. Srivastava, K. Greff, and J. Schmidhuber, “Training Very Deep Networks,” arXiv:1507.06228 [cs.LG], 2015.
3. K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” arXiv:1502.01852, 2015.
4. A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “FitNets: Hints for Thin Deep Nets,” arXiv:1412.6550, 2014.
