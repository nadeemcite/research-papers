# Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)  
**arXiv:** [1502.01852](https://arxiv.org/abs/1502.01852)  
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/delving-deep-into-rectifiers-he-init)
**Published:** 6 February 2015

## One-paragraph summary

This paper introduced two practical advances that made very deep rectifier networks trainable at scale. First, the authors proposed **Parametric ReLU (PReLU)**, a generalization of ReLU in which the slope of the negative half is no longer fixed at 0 (or 0.01 for Leaky ReLU) but is learned jointly with the rest of the model. PReLU adds only one extra parameter per channel and improves model fitting with almost no extra compute. Second, the paper derived **He initialization**, a theoretically grounded weight initialization that accounts for the fact that ReLU zeros-out half of its inputs. Instead of the Xavier/Glorot rule `Var(W)=1/fan_in`, He initialization uses `Var(W)=2/fan_in` for ReLU, so the variance of activations stays roughly constant across many layers. When combined with SPP-net-style architectures, PReLU-nets reached **4.94% top-5 error on ImageNet 2012**, the first result reported to surpass human-level performance on that benchmark.

## What problem does it solve

Imagine you are whispering a message through a very long line of people — each person can only pass on what they hear. If every person mumbles a little more quietly than the person before, by the time the message reaches the end it is so faint that nobody can hear it. That is what happens in a deep neural network when the weights are initialized too small: the signal **vanishes** before it reaches the output. If every person shouts a little louder than the one before, the message becomes deafening noise by the end. That is what happens when weights are initialized too large: the signal **explodes**.

Deep networks with ReLU activations were especially tricky because ReLU literally cuts off the negative half of each signal, which changes the average strength of the message at every layer. He et al. figured out exactly how loud each person should speak so the message stays the same volume all the way down the line. They also gave each person a small knob (PReLU) to decide how much of the "negative" part of the message to keep, instead of throwing it away by default. The result is that very deep networks can be trained from scratch without the signal dying or blowing up.

## Core idea and key method details

* **PReLU** is defined as `f(y_i) = max(0, y_i) + a_i * min(0, y_i)`. When `a_i=0` it reduces to ReLU; when `a_i` is learned it becomes PReLU. A channel-shared variant uses a single `a` per layer. PReLU is initialized at `a=0.25` and trained with plain SGD + momentum (no weight decay on `a`).
* **He initialization** is derived by requiring that the variance of a layer's output equals the variance of its input. For ReLU, because `E[x^2] = 1/2 Var[y]` after the rectifier, the variance recursion becomes `Var[y^L] = Var[y^1] * prod_l (1/2 * n_l * Var[w_l])`. Setting each factor to 1 gives `Var[w_l] = 2 / n_l`, i.e. a zero-mean Gaussian with standard deviation `sqrt(2/fan_in)`. For PReLU the factor generalizes to `2 / ((1 + a^2) * fan_in)`.
* The derivation is given for both forward propagation and back-propagation, showing that keeping either the forward signal or the backward gradient stable is sufficient for the other to remain stable as well.
* The authors trained extremely deep models (up to 30 weight layers) directly from scratch, whereas prior work either stalled or needed layer-wise pre-training with Xavier-style initialization.

## Influence

The He initialization rule became the de-facto default for ReLU networks in PyTorch (`torch.nn.init.kaiming_normal_`) and TensorFlow/Keras (`he_normal`). PReLU influenced later learned-activation research such as GELU, Swish, and Maxout, and the paper's analysis remains a standard reference for why deep networks need careful weight scaling.
