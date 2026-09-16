# Dropout — Improving Neural Networks by Preventing Co-adaptation of Feature Detectors

**Paper:** [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
**Authors:** Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov
**Year:** 2012

---

## Summary

Large neural networks trained on small datasets have a chronic problem: they memorize the training examples instead of learning general rules, so they perform poorly on new test data. This is called **overfitting**. Hinton et al. proposed a surprisingly simple fix: during training, randomly drop out (set to zero) roughly half of the neurons in the network for every training example. Each update trains a different "thinned" sub-network, which forces the network to learn robust features that work in many different contexts rather than relying on specific co-adapted combinations of neurons.

## The Core Idea

The breakthrough insight is that **neurons should not depend on each other too much**. In a standard network, a hidden neuron can become useful only because another specific neuron is present; this is "co-adaptation." Dropout breaks these co-adaptations by randomly deleting neurons on every forward pass. Because the network never knows which neurons will be available, each neuron must learn to detect a feature that is independently useful across a wide variety of internal contexts. At test time, all neurons are used but their outgoing weights are scaled down by the dropout probability, which approximates the effect of averaging over all the thinned networks.

## Key Method Details

### Training-Time Dropout
- For each training example, each hidden unit is kept active with probability `p` (typically `p = 0.5` for hidden layers, `p = 0.8` for input layers).
- A binary mask `r` is sampled from a Bernoulli distribution with probability `p` and multiplied element-wise with the layer's output: `y = r * f(Wx + b)`.
- During backpropagation, gradients flow only through the active units.

### Test-Time Weight Scaling
- At inference time, dropout is turned off (all units are active).
- To approximate averaging over many thinned networks, outgoing weights from a layer with dropout probability `p` are multiplied by `p`.
- For example, if hidden units were retained 50% of the time, their output weights are halved at test time.

### Why It Works
- **Ensemble effect:** Dropout trains an exponential number of smaller networks and approximately averages their predictions.
- **Robust features:** Each unit must be useful on its own, not just in a specific combination with other units.
- **Regularization:** It acts as a strong regularizer, reducing the need for other forms of regularization such as weight decay or early stopping.

### Training Setup (from the paper)
- Tested on MNIST, CIFAR-10, ImageNet, and speech recognition tasks.
- For MNIST: fully connected networks with ReLU/sigmoid activations.
- For ImageNet: AlexNet-style convolutional networks with dropout in fully connected layers.
- Optimizer: SGD with momentum, learning rate tuned per task.

## Why It Mattered

Dropout became one of the most widely used regularization techniques in deep learning. It is simple to implement, adds almost no computational cost, and dramatically improves generalization across vision, speech, and NLP tasks. The 2012 ImageNet win by AlexNet used dropout in the fully connected layers, helping establish deep learning as the dominant approach in computer vision. The follow-up JMLR paper (2014) provided a more thorough analysis and is one of the most cited deep-learning papers of all time.

## Influence Afterward

- **Batch Normalization** (Ioffe & Szegedy, 2015) — reduces internal covariate shift and, in some cases, reduces the need for dropout.
- **DropConnect** (Wan et al., 2013) — randomly drops weights instead of activations.
- **Spatial Dropout** (Tompson et al., 2015) — drops entire feature maps in convolutional networks.
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016) — reinterpreted dropout as approximate Bayesian inference for uncertainty estimation.
- **Stochastic Depth** (Huang et al., 2016) — randomly drops residual blocks during training.
- Dropout remains a standard baseline regularizer, especially for fully connected layers.