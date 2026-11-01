---
title: "Adam: Talk & Q&A"
parent: "003 Adam"
nav_order: 2
---
# TALK.md — Adam Optimizer

## Press coverage, blog posts, and official announcements

- The paper was published as a conference paper at **ICLR 2015** and is available on arXiv: https://arxiv.org/abs/1412.6980.
- **fast.ai blog** — "AdamW and Super-convergence is now the fastest way to train neural nets" (July 2018) by Sylvain Gugger and Jeremy Howard discusses the history of Adam and the AdamW fix: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html.
- **1-bit Adam paper** — Tang et al., "1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed" (2021), from the Microsoft DeepSpeed team, compresses Adam's communication for large-scale distributed training: https://arxiv.org/abs/2102.02888.
- **Hugging Face paper page** for Adam includes a concise abstract and links: https://huggingface.co/papers/1412.6980.

## Q&A (answers from the paper)

**Q1. What does "Adam" stand for, and what are its two moment estimates?**  
A (from the paper, Section 1 and Algorithm 1): The name comes from **adaptive moment estimation**. Adam keeps an exponential moving average of the gradient (the first moment, `m_t`) and of the elementwise squared gradient (the second moment, `v_t`), and uses them to give each parameter its own adaptive step size.

**Q2. Why is the bias-correction term necessary?**  
A (from the paper, Sections 2 and 3): `m_t` and `v_t` start at zero, so in the first steps they are biased toward zero, especially when the decay rates `β1`, `β2` are close to 1. Dividing by `(1 − β1^t)` and `(1 − β2^t)` removes this initialization bias; the correction goes to 1 as `t` grows.

**Q3. How does Adam differ from RMSProp?**  
A (from the paper, Section 5): RMSProp with momentum applies momentum to the rescaled gradient, while Adam builds its update directly from running averages of the first and second moments. RMSProp also has no bias correction, which matters most when `β2` is close to 1 (needed for sparse gradients): without it, steps can become very large and training often diverges (shown in Section 6.4).

**Q4. What are the default hyperparameters?**  
A (from the paper, Algorithm 1): `α = 0.001` (step size), `β1 = 0.9` and `β2 = 0.999` (decay rates of the two moment estimates), and `ε = 10⁻⁸` (a small constant that avoids division by zero). The paper calls these good defaults "for the tested machine learning problems".

**Q5. How big is each Adam step?**  
A (from the paper, Section 2.1): Roughly at most the step size `α`, whatever the scale of the gradient, because the update is the ratio `m̂_t / √v̂_t`. The paper describes this as a trust region around the current parameters. It makes the right scale of `α` fairly easy to guess, and the steps shrink automatically when the gradient is noisy (a low "signal-to-noise ratio").

## Common misconceptions or mistakes

- **"Adam never needs a learning-rate schedule."** It does; the default `α` may be too high for some models, and cosine or step decay often improves final performance.
- **"Adam always beats SGD."** Not universally true; on several benchmarks SGD with momentum matches or exceeds Adam when well tuned.
- **"Weight decay in Adam works the same as in SGD."** Standard Adam couples weight decay with the adaptive scaling. The paper that introduced **AdamW** ([Loshchilov & Hutter, "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101)) showed the two are not equivalent for Adam and that decoupled weight decay works better.
- **"Bias correction is a minor detail."** In the first few steps it prevents the optimizer from taking absurdly large steps; it is part of what makes Adam stable early in training.
- **"AdaMax is a separate optimizer from a later paper."** It comes from the same paper (Section 7.1): AdaMax replaces Adam's L2-based second-moment estimate with an L∞ norm, giving a simpler update that needs no bias correction for that term. It is used much less often than Adam.

## Sources

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015. https://arxiv.org/abs/1412.6980
- Gugger, S., & Howard, J. (2018). AdamW and Super-convergence is now the fastest way to train neural nets. fast.ai. https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
- Tang, H., Gan, S., Awan, A. A., Rajbhandari, S., Li, C., Lian, X., Liu, J., Zhang, C., & He, Y. (2021). 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed. https://arxiv.org/abs/2102.02888
- Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
