---
title: "He Init / PReLU: Talk & Q&A"
parent: "007 He Init / PReLU"
nav_order: 2
---
# TALK — Delving Deep into Rectifiers: Context, Discussion, and Q&A

**Paper:** [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (arXiv:1502.01852)](https://arxiv.org/abs/1502.01852)

## Verifiable press / blog coverage

* [The Morning Paper: "Optimisation and training techniques for deep learning"](https://blog.acolyer.org/2017/03/01/optimisation-and-training-techniques-for-deep-learning/) (Adrian Colyer, March 2017) — covers this paper in a post about the training-technique papers on the "top 100 awesome deep learning papers" list, explaining PReLU and the new initialization.
* [arXiv trackbacks for 1502.01852](https://arxiv.org/tb/1502.01852) — arXiv's list of blog posts that link to the paper.
* [PyTorch documentation for `torch.nn.init.kaiming_normal_`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) — cites this paper and says it is "also known as He initialization".
* [Keras documentation: `HeNormal` initializer](https://keras.io/api/layers/initializers/#henormal-class) — the Keras He initializers reference He et al., 2015.

## Q&A (answers from the paper)

**Q: What was the most surprising result in this paper?**  
A (from the paper, Abstract, Section 1 and Section 4): The best single model reached 5.71% top-5 error, better than all existing multi-model results. The multi-model result reached 4.94% top-5 test error on ImageNet 2012, a 26% relative improvement over GoogLeNet (6.66%). The authors say that, to their knowledge, this is the first result to surpass the reported human-level performance (5.1%). Their initialization also let them train very deep models (e.g., 30 weight layers) from scratch.

**Q: Is PReLU just a fancy Leaky ReLU?**  
A (from the paper, Section 2.1): Not quite. If the negative slope `a_i` is small and fixed, PReLU becomes Leaky ReLU. In PReLU the slope is learned, per channel or shared per layer. The learned slopes differ by layer: for example, the first conv layer learned slopes of 0.681 and 0.596, well above 0 (Table 1). On their 14-layer test model, PReLU gave a 1.2% top-1 gain over ReLU with nearly zero extra computation.

**Q: Why is the magic number `2` in He initialization?**  
A (from the paper, Section 2.2): ReLU sets the negative half of its input to zero, so `E[x²] = ½ Var[y]`. To keep the variance the same from layer to layer, each factor `½ n_l Var[w_l]` must be 1, which gives `Var[w_l] = 2 / n_l`. The linear ("Xavier") case has no factor of ½, so it uses `1 / n_l`.

**Q: Should I always use He initialization for every network?**  
A (general background, not from the paper): It is the usual default for ReLU-like activations. For PReLU the paper's formula becomes `2 / ((1 + a²) n_l)`. For tanh or sigmoid, Xavier/Glorot initialization is still the common choice, because the factor of 2 comes from the ReLU.

**Q: Did the paper invent ResNet?**  
A (general background, not from the paper): No. ResNet (also by Kaiming He and colleagues) came later, in December 2015. This paper showed that deep rectifier nets could be trained with a proper initialization; ResNet then trained even deeper nets by adding identity skip connections.

## Common misconceptions

1. **"He init is just Xavier init with a factor of 2."**  
   The factor is a consequence of the ReLU non-linearity, but the derivation is different: He init explicitly computes `E[x^2]` for a rectified distribution rather than assuming zero-mean activations.
2. **"PReLU causes huge overfitting."**  
   The paper reports the opposite. The extra parameter count is negligible (one scalar per channel or per layer), so it does not meaningfully increase model capacity.
3. **"This paper is only about ImageNet."**  
   ImageNet was the benchmark, but the two main ideas — PReLU and He init — are used across vision, speech, NLP, and scientific ML.

## Real citations

* He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. arXiv:1502.01852 [cs.CV].
* Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS 2010. (cited as the Xavier baseline).
* Russakovsky, O., et al. (2015). *ImageNet Large Scale Visual Recognition Challenge*. IJCV. (cited for the 5.1% human-level top-5 error benchmark.)
* Szegedy, C., et al. (2015). *Going Deeper with Convolutions*. CVPR 2015. (GoogLeNet, the 6.66% prior state of the art.)
* Nair, V., & Hinton, G. E. (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*. ICML 2010. (early ReLU usage.)
