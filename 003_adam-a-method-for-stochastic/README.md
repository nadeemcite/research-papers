# Adam — A Method for Stochastic Optimization

**Paper:** Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.  
**arXiv:** https://arxiv.org/abs/1412.6980
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/adam-a-method-for-stochastic-optimization)

## What problem does it solve?

Imagine you're blindfolded on a hilly trail and trying to find the lowest point in a valley. You feel the ground with your feet and take a step downhill. That's how neural networks learn — they feel the "slope" (gradient) and take a step. But the problem is: some parts of the hill are super steep and some are nearly flat. If you take the same-sized step everywhere, you'll overshoot on the steep parts and crawl on the flat parts. You need a smart way to take big steps on flat ground and tiny careful steps on steep ground.

Adam solves this by giving each parameter its own personal step size that automatically adjusts. It's like hiking with a smart pedometer that remembers two things: which direction you've been heading (momentum) and how bumpy the ground has been (step size scaling). If the ground is consistently sloping one way, Adam takes confident steps. If the ground is bouncing up and down, Adam takes tiny careful steps so it doesn't fall over. And it has a special trick to avoid being too cautious at the very start when it hasn't seen enough of the trail yet. This made Adam the go-to "engine" for training almost every modern AI model.

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
