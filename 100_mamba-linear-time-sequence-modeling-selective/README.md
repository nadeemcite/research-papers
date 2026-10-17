# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors:** Albert Gu, Tri Dao  
**arXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)  
**Published:** December 2023 (v2: May 2024)

## Summary

Mamba introduces a **selective state-space model (S6)** that closes the performance gap between efficient recurrent architectures and Transformers on language modeling. The key insight is that prior state-space models (S4 and its variants) were *linear-time-invariant* (LTI) — their parameters were fixed regardless of input, so they could not selectively propagate or forget information along the sequence. Mamba makes the SSM parameters (B, C, and the step size Δ) **input-dependent functions** of the current token, enabling content-based reasoning similar to attention while retaining O(N) linear-time inference. To make this efficient despite losing the convolutional computation that LTI models enjoyed, the authors design a **hardware-aware parallel scan algorithm** that fuses the discretization and scan steps in GPU SRAM, avoiding expensive HBM materialization. The result is a simplified, homogenous block architecture (no attention, no separate MLP blocks) that achieves 5× higher inference throughput than Transformers and scales linearly to million-length sequences.

## Core Idea

A continuous-time state-space model maps a 1-D input signal x(t) to output y(t) through a latent state h(t) ∈ ℝ^N:

- h′(t) = A·h(t) + B·x(t)  (state evolution)
- y(t) = C·h(t)              (output projection)

Discretized via Zero-Order Hold (ZOH), this becomes a recurrence:
- h_t = Ā·h_{t-1} + B̄·x_t
- y_t = C·h_t

where Ā = exp(Δ·A) and B̄ = (Δ·A)⁻¹(exp(Δ·A) − I)·Δ·B.

In **S4** (the predecessor), A, B, C, and Δ are *learned but fixed* parameters — the model is LTI and can be computed as either a recurrence (O(N) per step) or a global convolution (O(L log L) total). The limitation: the model treats every token identically; it cannot "look at" the input content to decide what to remember or forget.

**Mamba's selection mechanism (S6)** makes B, C, and Δ *functions of the input x_t*:
- B_t = s_B(x_t),  C_t = s_C(x_t),  Δ_t = τ_Δ(Parameter + s_Δ(x_t))

This breaks LTI (the model is now time-varying), so the convolutional mode no longer applies. But the recurrence becomes *selective*: at each step the model can choose to write new information into the state (large Δ, large B) or reset/forget (small Δ, small B). The parameter A remains a fixed structured matrix (initialized via HiPPO for long-range memory).

The **hardware-aware scan** overcomes the sequential nature of the recurrence by using a parallel scan algorithm (associative reduction) fused with the discretization step, keeping intermediate state in fast SRAM and only reading/writing inputs and outputs from HBM. This makes selective SSMs practical on modern GPUs.

The **Mamba block** combines this selective SSM with a gated MLP (SiLU/Swish activation, SwiGLU-style) in a single homogenous block: input → linear projection → conv1d → SiLU → selective SSM → gating → output. The block is repeated uniformly (no interleaving with attention or separate MLP blocks).

## What problem does it solve

Imagine you're reading a very long book — say, 500 pages — and someone asks you a question about something that happened on page 3. You'd need to somehow remember that early detail while reading through all 500 pages. 

Computers face the same problem when processing long texts. The most popular method today (the Transformer, which powers ChatGPT) works like this: for every word it reads, it looks back at *every single previous word*. If the text has 100,000 words, that's 100,000 × 100,000 = 10 billion lookbacks! This "quadratic" slowdown means Transformers become extremely slow and memory-hungry as texts get longer. It's like re-reading the entire book every time you reach a new word.

Older approaches called RNNs (recurrent neural networks) were faster — they kept a small "summary notebook" of everything seen so far and only updated it with each new word. But they had a problem too: they used the *same* notebook-updating rule for every word, whether the word was important or not. So they'd either forget crucial early details or get cluttered with useless information.

**Mamba solves both problems at once.** It keeps the RNN's fast "notebook" approach (linear time, no looking back at everything), but makes the notebook *smart*: for each new word, it actively decides whether this word is worth writing down in the notebook or should be ignored. It's like a reader who maintains a small summary but knows *which details to keep and which to skip* — remembering the crucial clue from page 3 while ignoring filler descriptions on page 200. This lets Mamba handle texts up to a million words long quickly, while matching or beating Transformers on quality.

## Key Method Details

- **Selection mechanism:** B, C, Δ are input-dependent (linear projections of x_t); A remains a fixed learnable parameter (HiPPO-initialized, diagonal structure).
- **Discretization:** Zero-Order Hold (ZOH): Ā = exp(Δ·A), B̄ = (Δ·A)⁻¹(exp(Δ·A) − I)·Δ·B.
- **Hardware-aware parallel scan:** Fuses discretization + scan in SRAM; avoids materializing (B, L, D, N)-shaped tensors in HBM.
- **Mamba block:** Linear → Conv1d → SiLU → SSM → gating → Linear (with residual connection and optional LayerNorm).
- **Simplifications in this notebook:** We implement the core selective SSM (S6) from scratch in PyTorch, train on toy sequence tasks (selective copy, induction-head-style), and benchmark speed/memory against a vanilla Transformer at increasing sequence lengths. We do not implement the full CUDA-fused scan (we use a PyTorch parallel scan / naive scan for clarity).

## Influence

Mamba has become one of the most influential papers of 2023-2024, sparking a wave of SSM-based architectures (Jamba, Vision Mamba, MambaByte, Mamba-2, etc.). It demonstrated that attention-free architectures can match Transformer scaling laws, challenging the dominance of attention-based models. The paper has been cited extensively in both academia and industry.

## Kaggle

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mamba-selective-state-space-model)
