# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Authors:** Sergey Ioffe, Christian Szegedy  
**arXiv:** [1502.03167](https://arxiv.org/abs/1502.03167)  
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/batch-normalization)
**Published:** 11 Feb 2015 (revised 2 Mar 2015)

Batch Normalization (BN) is a method that normalizes the inputs of every layer in a deep neural network so that, for each mini-batch during training, the activations have mean 0 and variance 1 before they enter the nonlinearity. Ioffe and Szegedy introduced BN to fight *internal covariate shift* — the phenomenon where the distribution of each layer’s inputs keeps changing as the parameters of all earlier layers are updated. By making normalization part of the network architecture and backpropagating through the mini-batch statistics, BN lets optimizers use much larger learning rates, relaxes the need for careful initialization, and often removes the need for Dropout. In their ImageNet experiments, a batch-normalized Inception variant reached the baseline accuracy with 14× fewer training steps and ultimately set a new state of the art (4.9% top-5 validation error with an ensemble).

## What problem does it solve

Imagine you are learning to bake cookies with a friend who keeps secretly changing the measuring cups while you mix the dough. One batch uses a 1-cup scoop, the next uses a 1/2-cup scoop, and you never know which one is in the bowl. You would have to relearn the recipe every time, and progress would be painfully slow. In a deep neural network, each layer is like you, and the layers before it are like the friend changing the measuring cups. As the earlier layers learn, the numbers they pass to the next layer change their average size and spread, so every downstream layer must keep readjusting. Batch Normalization solves this by placing a “standard ruler” between layers: it measures the numbers in one mini-batch, shifts them so their average is 0 and their spread is 1, and then lets the network learn a gentle rescale and shift on top of that. With the ruler in place, later layers see a stable recipe, training can go faster with bigger steps, and activations stay in the healthy range of the activation function instead of getting stuck at the extremes.

## Core idea and key method details

- **Internal covariate shift:** during SGD, the input distribution to every internal sub-network changes because upstream parameters change. BN fixes the first two moments (mean and variance) of each layer’s input distribution.
- **Per-activation normalization:** for a layer input `x` of dimension `d`, each scalar feature is normalized independently over the mini-batch `B`:
  ```
  μ_B = mean(x_i for i in B)
  σ²_B = variance(x_i for i in B)
  x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)
  y_i = γ · x̂_i + β
  ```
  `γ` and `β` are learned per feature and let the transform represent the identity when needed.
- **Differentiable transform:** the paper derives gradients for `x_i`, `γ`, and `β` so the mini-batch statistics participate fully in backpropagation, preventing the parameter-explosion problem that occurs when normalization is applied outside the gradient step.
- **Convolutional layers:** for convolutions, BN jointly normalizes all activations of a feature map across both the mini-batch and all spatial locations, using one `(γ, β)` pair per feature map.
- **Training vs. inference:** during training, statistics come from each mini-batch. At inference, BN uses population estimates:
  ```
  E[x] = E_B[μ_B]
  Var[x] = (m / (m - 1)) · E_B[σ²_B]
  ```
  which are collected as moving averages and folded into a single linear transform at test time.
- **Side effects:** BN stabilizes gradients with respect to parameter scale, allows higher learning rates, makes saturating nonlinearities usable again, and acts as a regularizer by adding mini-batch-dependent noise to each example.

## Influence

Batch Normalization became a near-universal building block in deep learning. ResNets, DenseNets, MobileNets, EfficientNets, Transformers, and nearly every modern vision or language model use some form of normalization. The paper is one of the most cited in machine learning (over 47,000 citations on Semantic Scholar) and helped make very deep networks trainable without hand-crafted initialization tricks. It also spurred a family of successors — Layer Normalization, Instance Normalization, Group Normalization, and Switchable Normalization — each adapting the same moment-normalization idea to different data layouts and batch constraints.
