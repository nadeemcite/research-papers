# Adam — A Method for Stochastic Optimization

**Paper:** Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.  
**arXiv:** https://arxiv.org/abs/1412.6980

## What problem does it solve?

Training modern machine-learning models means repeatedly adjusting thousands or millions of parameters using noisy gradients computed on mini-batches. Vanilla stochastic gradient descent (SGD) uses one global learning rate, so it takes forever on some parameters and overshoots on others. The paper asks: can each parameter get its own adaptive learning rate, while keeping memory tiny and the implementation simple? Adam answers yes by combining the idea of momentum with the per-parameter scaling ideas behind AdaGrad and RMSProp.

## The core insight

Adam computes a **running estimate of the first moment** (mean gradient, i.e. momentum) and a **running estimate of the second moment** (mean squared gradient, i.e. per-parameter scale) of the gradient. It then divides the momentum step by the square root of the second moment, so parameters that historically bounce around get tiny steps and parameters with a consistent direction get larger steps. The clever twist is a **bias-correction term** in the first few iterations: because the moving averages are initialized at zero, early estimates are biased toward zero; Adam divides them by a correction factor that quickly converges to 1.

## Key method details (for the code)

The algorithm, for parameter vector `θ`, gradient `g_t`, and step `t`:

1. Compute gradient: `g_t = ∇f_t(θ_{t-1})`
2. Update biased first-moment estimate: `m_t = β1 · m_{t-1} + (1 − β1) · g_t`
3. Update biased second-moment estimate: `v_t = β2 · v_{t-1} + (1 − β2) · g_t²`  (elementwise)
4. Bias-correct:
   - `m̂_t = m_t / (1 − β1^t)`
   - `v̂_t = v_t / (1 − β2^t)`
5. Update parameters: `θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)`

Recommended defaults: `α = 0.001`, `β1 = 0.9`, `β2 = 0.999`, `ε = 10⁻⁸`.

Adam is invariant to diagonal rescaling of the gradients, works with sparse gradients, non-stationary objectives, and naturally anneals step sizes as `v_t` accumulates history.

## Why it mattered

Adam became the default optimizer for a huge swath of deep learning — transformers, GANs, variational autoencoders, recommendation models — because it usually trains fast with little hyperparameter tuning. It also inspired later variants such as **AdamW** (which fixes weight-decay interaction) and **AMSGrad**. Even today, with newer flavors like `β1 = β2` modern variants, the Adam family remains the dominant optimizer in large-scale neural-network training.
