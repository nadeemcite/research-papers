---
title: "100 Mamba"
nav_order: 101
permalink: "/100_mamba-linear-time-sequence-modeling-selective/"
---
# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mamba-selective-state-space-model)

- **Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **Authors:** Albert Gu, Tri Dao
- **Lab, year:** Carnegie Mellon University and Princeton University, 2023 (v2: May 2024)

## Summary

Mamba introduces a **selective state-space model (S6)** that closes much of the gap between efficient recurrent-style models and Transformers on language modeling. Earlier state-space models (S4 and its variants) were *linear time-invariant* (LTI): their parameters were the same for every token, so they could not choose what to keep or forget based on the content.

Mamba makes the SSM parameters B, C and the step size Δ **functions of the current token**. This gives content-based selection, a bit like attention, while keeping linear time in the sequence length. Because this removes the fast convolution trick of LTI models, the authors design a **hardware-aware parallel scan** that keeps the large state in fast GPU SRAM instead of slow GPU memory (HBM).

The result is a simple block design with no attention and no separate MLP blocks. The paper reports 5× higher inference throughput than Transformers, linear scaling in sequence length, and quality that keeps improving on real data up to million-length sequences. Mamba-3B beats Transformers of the same size and matches Transformers twice its size.

## What Problem Does It Solve

Imagine you're reading a very long book — say, 500 pages — and someone asks you a question about something that happened on page 3. You'd need to somehow remember that early detail while reading through all 500 pages.

Computers face the same problem with long texts. The most popular method today (the Transformer, which powers ChatGPT) looks back at *every single previous word* for each new word. If the text has 100,000 words, that's 100,000 × 100,000 = 10 billion lookbacks! This "quadratic" cost makes Transformers slow and memory-hungry on long texts. It's like re-reading the whole book every time you reach a new word.

Older approaches called RNNs (recurrent neural networks) were cheaper. They kept a small "summary notebook" and updated it with each new word. But earlier state-space models used the *same* updating rule for every word, important or not, so they could either forget key details or fill up with useless ones.

**Mamba keeps the cheap notebook but makes it smart.** For each new word, it decides how much to write down and how much to ignore. It's like a reader who keeps a short summary but knows *which details to keep* — the clue on page 3 — and which filler to skip on page 200.

## Core Idea

A continuous-time state-space model maps a 1-D input signal x(t) to output y(t) through a hidden state h(t) ∈ ℝ^N:

- h′(t) = A·h(t) + B·x(t)  (state evolution)
- y(t) = C·h(t)              (output projection)

Discretized with a Zero-Order Hold (ZOH), this becomes a recurrence:
- h_t = Ā·h_{t-1} + B̄·x_t
- y_t = C·h_t

where Ā = exp(Δ·A) and B̄ = (Δ·A)⁻¹(exp(Δ·A) − I)·Δ·B.

In **S4** (the predecessor), A, B, C and Δ are *learned but fixed* parameters. The model is LTI and can be computed either as a recurrence or as one global convolution. The limitation: it treats every token the same way and cannot "look at" the content to decide what to remember or forget.

**Mamba's selection mechanism (S6)** makes B, C and Δ *functions of the input x_t*:
- B_t = s_B(x_t),  C_t = s_C(x_t),  Δ_t = τ_Δ(Parameter + s_Δ(x_t))

This makes the model time-varying, so the convolution mode no longer applies. But the recurrence becomes *selective*. In the paper's words (Section 3.5.2), a large Δ resets the state and focuses on the current input, while a small Δ keeps the state and ignores the current input. A stays a fixed (not input-dependent) learned parameter; the paper uses a real diagonal initialization (S4D-Real) based on HiPPO theory.

The **hardware-aware scan** (Section 3.3) computes the recurrence with a parallel scan, fuses it with the discretization, and keeps the intermediate states in fast SRAM, reading and writing only inputs and outputs from HBM. It also recomputes states in the backward pass instead of storing them.

The **Mamba block** (Section 3.4) merges the H3 block and the gated MLP into one block: input → linear projection → conv1d → SiLU → selective SSM → multiply by a SiLU gate → linear projection. The same block is repeated, with no attention and no separate MLP blocks.

## Key Method Details

- **Selection mechanism:** B, C and Δ are input-dependent (linear projections of x_t); A is fixed with respect to the input (learned, diagonal, S4D-Real initialization).
- **Discretization:** Zero-Order Hold (ZOH): Ā = exp(Δ·A), B̄ = (Δ·A)⁻¹(exp(Δ·A) − I)·Δ·B.
- **Hardware-aware parallel scan:** fuses discretization and scan in SRAM and avoids materializing (B, L, D, N)-shaped tensors in HBM. The paper's scan is faster than FlashAttention-2 beyond sequence length 2K, and up to 20-40× faster than a standard PyTorch scan (Section 4.5).
- **Mamba block:** Linear → Conv1d → SiLU → SSM → gating → Linear, with a residual connection and an optional normalization layer.
- **Synthetic tasks (Section 4.1):** the selective SSM solves Selective Copying, and on Induction Heads Mamba generalizes perfectly to million-length sequences, 4000× longer than in training.

### Simplifications in This Notebook

- A small selective SSM written in plain PyTorch, with a **naive Python loop** over time steps instead of the fused CUDA parallel scan
- Simplified ZOH for B̄ (`B̄ = Δ·B`), diagonal A
- Tiny models: d_model 64, state size 16, 2 layers (Mamba 35,072 parameters, Transformer 101,120)
- Toy tasks only (selective copy and induction heads, vocab 16, length 64), trained for 50 epochs of 10 batches each
- An `S4Layer` (LTI baseline) is defined but never trained, so there is no S4 vs S6 comparison

### What the Notebook Result Shows

Training loss only (no accuracy is measured):

- Selective copy: Mamba 0.5381, Transformer 0.6373
- Induction heads: Mamba 0.1131, Transformer 0.1141

Both models end at similar losses, and without accuracy we cannot say that either task is "solved".

**Speed:** this naive-loop Mamba was **much slower** than the Transformer at every length tested: 370.9 ms vs 5.0 ms at L=256 and 5858.1 ms vs 210.0 ms at L=4096 (28× to 131× slower). Mamba's time roughly doubled with each doubling of length (linear), while the Transformer's grew about 3.9× from 2048 to 4096 (closer to quadratic), but it was still far faster here. The real speed-up in the paper comes from the fused CUDA scan, which this notebook does not have.

**Memory:** this part matches the paper. From L=512 on, Mamba used less peak memory, and at L=4096 it used 537.0 MB vs 5496.4 MB for the Transformer (about 10× less), because there is no L × L attention matrix. The plots in the notebook were not displayed in the saved run.

## Influence

Mamba started a wave of SSM-based architectures, such as Jamba (Mamba mixed with attention), Vision Mamba, MambaByte and Mamba-2. In the paper's scaling-law tests (Section 4.2), Mamba was the first attention-free model to match a strong modern Transformer recipe ("Transformer++"), which made SSMs a serious alternative to attention.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/100_mamba-linear-time-sequence-modeling-selective/solution.ipynb) | From-scratch PyTorch selective SSM (S6) and Mamba block, trained on toy tasks, with a speed and memory benchmark against a small Transformer |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Blog posts, a talk with an author, Q&A (answers from the paper), and common misconceptions about the paper |
