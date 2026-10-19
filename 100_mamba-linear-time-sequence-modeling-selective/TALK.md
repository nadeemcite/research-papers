---
title: "Mamba: Talk & Q&A"
parent: "100 Mamba"
nav_order: 2
---
# TALK.md — Mamba: Context, Discussion, and Q&A

**Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)

## Verifiable Press/Blog Coverage

1. [The Gradient: "Mamba Explained"](https://thegradient.pub/mamba-explained/) (27 March 2024) — a 20-minute explainer covering state space models, a "Temple Run" analogy for compressing the state, discretization and the selective scan.

2. [Maarten Grootendorst: "A Visual Guide to Mamba and State Space Models"](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) (19 February 2024) — an illustrated, step-by-step guide from SSMs to the selective scan.

3. [Hugging Face Transformers documentation: Mamba](https://huggingface.co/docs/transformers/model_doc/mamba) — the official model page for Mamba in the Transformers library.

4. [Official code: state-spaces/mamba on GitHub](https://github.com/state-spaces/mamba) — the authors' implementation, including the fused selective-scan kernels.

5. [Tri Dao and Albert Gu: "State Space Duality (Mamba-2) Part I - The Model"](https://tridao.me/blog/2024/mamba2-part1-model/) (31 May 2024) — the authors' own blog post on the follow-up, Mamba-2.

## Talks

1. [Interconnects: "Interviewing Tri Dao and Michael Poli of Together AI on the future of LLM architectures"](https://www.youtube.com/watch?v=OFFHiJzPpCQ) (21 December 2023) — a recorded conversation with Tri Dao and Michael Poli; chapters include "What is Mamba" and "Mamba hardware optimization". Written intro: [State-space LLMs: Do we need Attention?](https://www.interconnects.ai/p/llms-beyond-attention)

## Q&A (answers from the paper)

**Q: Why was linear time invariance (LTI) the key limitation of earlier SSMs?**
A (from the paper, Section 3.1): The authors see sequence modeling as compressing context into a small state. LTI models use the same dynamics for every token, so they cannot pick out relevant tokens. They can solve the plain Copying task by just counting time, but fail on Selective Copying, where the spacing between important tokens is random, and on Induction Heads, which needs content-based recall.

**Q: Why wouldn't a naive implementation of the selective SSM be fast?**
A (from the paper, Section 3.3): With input-dependent parameters, the model can no longer use a global convolution. A plain recurrence would also have to store a state of shape (B, L, D, N) in slow GPU memory (HBM). Mamba's kernel loads the parameters into fast SRAM, does the discretization and the scan there, writes back only the (B, L, D) outputs, and recomputes the states in the backward pass instead of storing them.

**Q: How does Mamba relate to RNN gating?**
A (from the paper, Section 3.5.1, Theorem 1): With N = 1, A = −1, B = 1 and a softplus Δ, the selective SSM reduces to the classic gated RNN update h_t = (1 − g_t) h_{t−1} + g_t x_t. So RNN gates are a special case of the selection mechanism, and Δ plays the role of the gate.

**Q: What are the limitations of Mamba?**
A (from the paper, Section 5): The authors name three. "No free lunch": selection helps on discrete data like text and DNA, but can hurt on continuous signals such as audio, where LTI SSMs do well. It is still open whether SSMs have the same "affordances" as Transformers (fine-tuning, prompting, in-context learning, RLHF, quantization). And the tests are limited to small models, below the size of most strong open-source LLMs.

**Q: How fast is Mamba in practice?**
A (from the paper, Section 4.5): The fused scan is faster than FlashAttention-2 beyond sequence length 2K and up to 20-40× faster than a standard PyTorch scan. Mamba has 4-5× higher inference throughput than a Transformer of similar size, because it needs no KV cache and can use much larger batches.

## Common Misconceptions

1. **"Mamba replaces attention entirely."** — Mamba removes attention from its architecture, but it doesn't claim attention is never useful. The paper discusses a "continuous-discrete spectrum" (Section 5): selection helps on discrete data like text but can hurt on continuous signals like audio. Hybrid architectures (e.g., Jamba) later combined Mamba blocks with attention blocks.

2. **"Mamba is just a fancy RNN."** — While Mamba uses recurrence, the selective SSM is fundamentally different from traditional RNNs (LSTMs, GRUs). It comes from continuous-time SSMs with a structured (diagonal, HiPPO-based) A matrix and a discretization step, and it is trained with a parallel scan. The paper also shows that classic RNN gating is a special case of its selection mechanism (Section 3.5.1).

3. **"Mamba is faster than Transformers at all sequence lengths."** — Not for every setup. In the paper, the fused scan beats FlashAttention-2 only beyond sequence length 2K (Section 4.5). And speed depends on the implementation: the naive Python-loop version in this series' notebook is 28× to 131× slower than a small Transformer at every length it tested (256 to 4096).

4. **"The A matrix is also input-dependent."** — In the original Mamba paper, A remains a *fixed* learnable parameter (only B, C, and Δ are input-dependent). Making A input-dependent would require more complex discretization. (Mamba-2 and subsequent work explored variants.)

## Real Citations

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. NeurIPS 2020.
- Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). *On the Parameterization and Initialization of Diagonal State Space Models*. NeurIPS 2022. (S4D)
- Fu, D. Y., Dao, T., et al. (2023). *Hungry Hungry Hippos: Towards Language Modeling with State Space Models*. ICLR 2023. (H3)
- Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. (Mamba-2)
